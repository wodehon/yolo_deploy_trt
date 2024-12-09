#include "yolov5_trt_ros.h"

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

YOLOv5::YOLOv5(const std::string& engine_name, ros::NodeHandle& nh) 
    : runtime_(nullptr), engine_(nullptr), context_(nullptr), cpu_output_buffer_(nullptr) {
    // Deserialize engine
    deserializeEngine(engine_name);

    // Prepare CUDA buffers
    prepareBuffers();

    // Get topic name from parameter or use default value
    std::string image_topic;
    nh.param<std::string>("image_topic", image_topic, "/camera/color/image_raw");

    // Initialize ROS subscribers and publishers
    image_sub_ = nh.subscribe(image_topic, 1, &YOLOv5::imageCallback, this);
    detection_pub_ = nh.advertise<yolo_deploy_trt::Detections2D>("/yolov5/detections", 1);
    img_res_pub_ = nh.advertise<sensor_msgs::Image>("/yolov5/detImg", 1);

    // Initialize CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Initialize filter
    // previous_ = {0.0f, 0.0f, 0.0f, 0.0f};
    // velocity_ = {0.0f, 0.0f, 0.0f, 0.0f};
    // filter_.bbox[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // filter_.conf = 0.0f;
    // filter_.class_id = 0.0f;
    // for (int i = 0; i < 32; ++i) {
    //     filter_.mask[i] = 0.0f;
    // }
    filter_ = {};

    // OneEuroFilter f(frequency, mincutoff, beta, dcutoff);
}

YOLOv5::~YOLOv5() {
    // Release resources
    cudaStreamDestroy(stream_);
    CUDA_CHECK(cudaFree(gpu_buffers_[0]));
    CUDA_CHECK(cudaFree(gpu_buffers_[1]));
    delete[] cpu_output_buffer_;
    cuda_preprocess_destroy();

    // Destroy TensorRT objects
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}

void YOLOv5::deserializeEngine(const std::string& engine_name) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        ROS_ERROR("Failed to read engine file: %s", engine_name.c_str());
        throw std::runtime_error("Failed to read engine file");
    }

    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    file.read(serialized_engine, size);
    file.close();

    runtime_ = createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(serialized_engine, size);
    context_ = engine_->createExecutionContext();
    delete[] serialized_engine;
}

void YOLOv5::prepareBuffers() {
    assert(engine_->getNbBindings() == 2);
    const int input_index = engine_->getBindingIndex(kInputTensorName);
    const int output_index = engine_->getBindingIndex(kOutputTensorName);
    assert(input_index == 0);
    assert(output_index == 1);

    CUDA_CHECK(cudaMalloc((void**)&gpu_buffers_[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&gpu_buffers_[1], kBatchSize * kOutputSize * sizeof(float)));
    cpu_output_buffer_ = new float[kBatchSize * kOutputSize];
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

void YOLOv5::infer(const cv::Mat& image, std::vector<Detection>& results) {
    // Preprocess the image
    std::vector<cv::Mat> img_batch = {image};
    cuda_batch_preprocess(img_batch, gpu_buffers_[0], kInputW, kInputH, stream_);

    // // Run inference
    // context_->enqueue(kBatchSize, (void**)gpu_buffers_, stream_, nullptr);

    // for enqueueV2
    // Set binding dimensions for dynamic input
    int inputIndex = engine_->getBindingIndex(kInputTensorName); // 输入绑定名称
    nvinfer1::Dims inputDims = context_->getBindingDimensions(inputIndex);
    inputDims.d[0] = kBatchSize;  // Batch size，通常是 1
    inputDims.d[1] = 3;           // 通道数（RGB）
    inputDims.d[2] = kInputH;     // 输入高度（例如 640）
    inputDims.d[3] = kInputW;     // 输入宽度（例如 640）
    context_->setBindingDimensions(inputIndex, inputDims);

    // Run inference
    context_->enqueueV2((void**)gpu_buffers_, stream_, nullptr);

    // Copy results to host
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_, gpu_buffers_[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // Apply NMS
    // batch_nms(results, cpu_output_buffer_, 1, kOutputSize, kConfThresh, kNmsThresh); // std::vector<Detection>& results <-> std::vector<std::vector<Detection>>& results
    nms(results, cpu_output_buffer_, kConfThresh, kNmsThresh);
}

void YOLOv5::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat img = cv_ptr->image;

        // Perform inference
        std::vector<Detection> results;
        auto start = std::chrono::system_clock::now();
        infer(img, results);
        auto end = std::chrono::system_clock::now();
        // std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        ROS_INFO_STREAM("results:" << formatResults(results) << 
            std::endl << "inference time: " << 
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl);
        if (results.size()!=0){
            lowPassFilter(filter_,results[0],0.5);
        }

        // Publish results (as an example, we publish as JSON string)
        // std_msgs::String detection_msg;
        // detection_msg.data = "Detections: " + std::to_string(results.size());
        // detection_pub_.publish(detection_msg);

        publish_detections(results);

        // Optionally, draw and save the results
        draw_bbox(img, results);
        cv::Mat resized_img;
        // cv::resize(img, resized_img, cv::Size(640, 480));
        cv::resize(img, resized_img, cv::Size(), 0.3, 0.3); // 宽和高都缩小 50%
        // cv::imshow("YOLOv5 Detection", resized_img);
        // cv::waitKey(1); 
        // cv::imwrite("result.jpg", img);
        publish_detected_image(resized_img);

    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void YOLOv5::publish_detected_image(cv::Mat& img) {
    // 将OpenCV图像转换为ROS图像消息
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, img);
    this->img_res_pub_.publish(cv_img.toImageMsg());
}

void YOLOv5::publish_detections(const std::vector<Detection>& results) {
    yolo_deploy_trt::Detections2D msg;
    msg.header.stamp = ros::Time::now();

    for (const auto& det : results) {
        yolo_deploy_trt::Detection2D detection;
        detection.confidence = det.conf;
        detection.class_id = det.class_id;

        detection.box_min.x = det.bbox[0]-0.5*det.bbox[2];
        detection.box_min.y = det.bbox[1]-0.5*det.bbox[3];
        detection.box_max.x = det.bbox[0]+0.5*det.bbox[2];
        detection.box_max.y = det.bbox[0]+0.5*det.bbox[2];

        msg.detections.push_back(detection);
    }

    this->detection_pub_.publish(msg);
}
