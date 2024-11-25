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

    // Initialize ROS subscribers and publishers
    image_sub_ = nh.subscribe("/camera/color/image_raw", 1, &YOLOv5::imageCallback, this);
    detection_pub_ = nh.advertise<std_msgs::String>("/yolov5/detections", 1);

    // Initialize CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);
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

void YOLOv5::infer(const cv::Mat& image, std::vector<std::vector<Detection>>& results) {
    // Preprocess the image
    std::vector<cv::Mat> img_batch = {image};
    cuda_batch_preprocess(img_batch, gpu_buffers_[0], kInputW, kInputH, stream_);

    // Run inference
    context_->enqueue(kBatchSize, (void**)gpu_buffers_, stream_, nullptr);

    // Copy results to host
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_, gpu_buffers_[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // Apply NMS
    batch_nms(results, cpu_output_buffer_, 1, kOutputSize, kConfThresh, kNmsThresh);
}

void YOLOv5::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image = cv_ptr->image;

        // Perform inference
        std::vector<std::vector<Detection>> results;
        infer(image, results);

        // Publish results (as an example, we publish as JSON string)
        std_msgs::String detection_msg;
        detection_msg.data = "Detections: " + std::to_string(results[0].size());
        detection_pub_.publish(detection_msg);

        // Optionally, draw and save the results
        std::vector<cv::Mat> img_batch = {image};
        draw_bbox(img_batch, results);
        cv::imwrite("result.jpg", image);

    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}
