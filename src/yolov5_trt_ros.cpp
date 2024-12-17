#include "yolov5_trt_ros.h"
#include "utils_transform.h" // 新增的转换函数头文件

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

YOLOv5::YOLOv5(const std::string& engine_name, ros::NodeHandle& nh) 
    : runtime_(nullptr), engine_(nullptr), context_(nullptr), cpu_output_buffer_(nullptr)
{
    nh.param<std::string>("engine_file", engine_file_, "yolov5.engine");
    nh.param<std::string>("image_topic", image_topic_, "/camera/color/image_raw");

    deserializeEngine(engine_file_);
    prepareBuffers();

    image_sub_ = nh.subscribe(image_topic_, 1, &YOLOv5::imageCallback, this);
    depth_sub_ = nh.subscribe("/camera/aligned_depth_to_color/image_raw",1,&YOLOv5::depthCallback,this);
    camera_info_sub_ = nh.subscribe("/camera/color/camera_info",1,&YOLOv5::cameraInfoCallback,this);
    pose_sub_ = nh.subscribe("/mavros/local_position/pose",1,&YOLOv5::poseCallback,this);

    detection_pub_ = nh.advertise<vision_msgs::Detection2DArray>("/yolov5/detections", 1);
    img_res_pub_ = nh.advertise<sensor_msgs::Image>("/yolov5/detImg", 1);

    pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/move_base_simple/goal",1);
    tri_pub_ = nh.advertise<std_msgs::Bool>("/insulator_pub_flag",1);
    tuning_pub_ = nh.advertise<std_msgs::Int8>("/yolov5/tuning",1);
    error_pub_ = nh.advertise<std_msgs::Float32>("/yolov5/error",1);
    discover_pub_ = nh.advertise<std_msgs::Bool>("/discover",1);
    hover_pub_ = nh.advertise<std_msgs::Bool>("/reach",1);

    cuda_preprocess_init(kMaxInputImageSize);
}

YOLOv5::~YOLOv5() {
    cudaStreamDestroy(stream_);
    CUDA_CHECK(cudaFree(gpu_buffers_[0]));
    CUDA_CHECK(cudaFree(gpu_buffers_[1]));
    delete[] cpu_output_buffer_;
    cuda_preprocess_destroy();

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
    std::vector<cv::Mat> img_batch = {image};
    cuda_batch_preprocess(img_batch, gpu_buffers_[0], kInputW, kInputH, stream_);

    // // Run inference
    context_->enqueue(kBatchSize, (void**)gpu_buffers_, stream_, nullptr);

    // for enqueueV2
    // Set binding dimensions for dynamic input
    // int inputIndex = engine_->getBindingIndex(kInputTensorName); // 输入绑定名称
    // nvinfer1::Dims inputDims = context_->getBindingDimensions(inputIndex);
    // inputDims.d[0] = kBatchSize;  // Batch size，通常是 1
    // inputDims.d[1] = 3;           // 通道数（RGB）
    // inputDims.d[2] = kInputH;     // 输入高度（例如 640）
    // inputDims.d[3] = kInputW;     // 输入宽度（例如 640）
    // context_->setBindingDimensions(inputIndex, inputDims);

    // // Run inference
    // context_->enqueueV2((void**)gpu_buffers_, stream_, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_, gpu_buffers_[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    nms(results, cpu_output_buffer_, kConfThresh, kNmsThresh);
}

void YOLOv5::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (!camera_info_received_ || !depth_received_) {
        ROS_INFO_THROTTLE(1,"Waiting for camera info and depth image...");
        return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat img = cv_ptr->image;
    std::vector<Detection> results;
    auto start = std::chrono::system_clock::now();
    infer(img, results);
    auto end = std::chrono::system_clock::now();

    ROS_INFO_STREAM("Detections: " << results.size() <<
                    ", inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms");

    handleDetections(img, results);
    draw_bbox(img, results);
    publish_detected_image(img);
}

void YOLOv5::depthCallback(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
    depth_img_ = cv_ptr->image;
    depth_received_ = true;
}

void YOLOv5::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    cv::Mat K_ = cv::Mat(3, 3, CV_64F, (void*)msg->K.data()).clone();
    cv::Mat D_ = cv::Mat(msg->D).clone();
    camera_info_received_ = true;
}

void YOLOv5::poseCallback(const geometry_msgs::PoseStampedConstPtr &msg) {
    current_pose_ = *msg;
    pose_received_ = true;
}

void YOLOv5::publish_detected_image(cv::Mat& img) {
    // 缩小图像以发布结果示例
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(), 0.3, 0.3);

    // cv::imshow("Display Window", resized_img);
    // cv::waitKey(2);

    std_msgs::Header header;
    header.stamp = ros::Time::now();
    cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, resized_img);
    img_res_pub_.publish(cv_img.toImageMsg());
}

void YOLOv5::publish_detections(const std::vector<Detection>& results) {
    vision_msgs::Detection2DArray msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "camera";

    for (auto &det : results) {
        vision_msgs::Detection2D detection;
        detection.bbox.center.x = det.bbox[0];
        detection.bbox.center.y = det.bbox[1];
        detection.bbox.size_x = det.bbox[2];
        detection.bbox.size_y = det.bbox[3];
        // 置信度和类别ID可在detection.results中填充
        vision_msgs::ObjectHypothesisWithPose hyp;
        hyp.id = (int)det.class_id;
        hyp.score = det.conf;
        detection.results.push_back(hyp);
        msg.detections.push_back(detection);
    }

    detection_pub_.publish(msg);
}

// 将Python节点逻辑整合到C++的函数中
void YOLOv5::handleDetections(cv::Mat &frame, const std::vector<Detection>& results) {
    // 根据python逻辑，我们需要：  
    // 1. 分类对象：tower/insulator等（需要类别名称映射，假设0: tower, 1: insulator）
    // 2. 对det进行处理，获得中心像素坐标(u,v)
    // 3. 从depth_img_中取depth
    // 4. 利用px2xy还原相机坐标，并转换到机体和世界坐标系
    // 5. 根据python中的逻辑进行tuning、trigger等判断
    // （下面的逻辑只是近似的移植和简化，需根据实际情况修改）

    int width = frame.cols;
    int height = frame.rows;

    // 发布用到的消息
    std_msgs::Bool tri_msg;
    tri_msg.data = trigger_;
    std_msgs::Int8 tuning_msg;
    tuning_msg.data = 0;
    std_msgs::Float32 error_msg;
    error_msg.data = 0.0f;
    std_msgs::Bool discover_msg;
    std_msgs::Bool hover_msg;
    hover_msg.data = false;

    bool discover = false; // 是否发现物体(tower)
    // 机体姿态
    if (!pose_received_) {
        ROS_INFO_STREAM("Waiting for uav_pose...");
        return;
    }
    Eigen::Vector3f trans(current_pose_.pose.position.x, current_pose_.pose.position.y, current_pose_.pose.position.z);
    Eigen::Matrix3f R_b2w = quatToRot(current_pose_.pose.orientation.w,
                                      current_pose_.pose.orientation.x,
                                      current_pose_.pose.orientation.y,
                                      current_pose_.pose.orientation.z).transpose(); // 转置求逆

    // 简单的类别名称假设：0->tower, 1->insulator
    // 实际可根据你的模型类别进行修改
    for (auto &det : results) {
        int class_id = (int)det.class_id;
        float conf = det.conf;
        float cx = det.bbox[0];
        float cy = det.bbox[1];
        float w = det.bbox[2];
        float h = det.bbox[3];

        if (cx<0 || cy<0 || cx>=width || cy>=height) continue;

        uint16_t d_val = depth_img_.at<uint16_t>((int)cy,(int)cx);
        float depth_m = (d_val == 0) ? MAX_Z_ * scale_ : d_val * scale_;

        Eigen::Vector3f point_cam = px2xy(cx, cy, K_, D_, depth_m);
        Eigen::Vector3f point_body = camtobody(point_cam);
        Eigen::Vector3f point_world = bodytoworld(point_body, trans, R_b2w);

        float wh_ratio = (float)w/(float)h;

        if (class_id == 0 && conf > 0.7 && !trigger_) { // tower逻辑
            error_msg.data = (width/2.0 - cx)/width;
            discover = results.empty(); 
            // ros::Time now = ros::Time::now();

            // tuning逻辑:
            if(!triggertest_){
                if (cx < width/2 && tuning_msg.data != -1) {
                    tuning_msg.data = 1;
                } else if (cx > width/2 && tuning_msg.data != 1) {
                    tuning_msg.data = -1;
                } else {
                    tuning_msg.data = 0;
                }
                count_++;
            }

            if (tuning_msg.data == 0 && count_ > 10) {
                triggertest_ = true;
                // 发布一个预设的goal
                geometry_msgs::PoseStamped pose_msg;
                pose_msg.header.stamp = ros::Time::now();
                pose_msg.header.frame_id = "map";
                // 此处使用python里给的固定坐标作为示例
                pose_msg.pose.position.x = -0.9407164;
                pose_msg.pose.position.y = 0.7008502;
                pose_msg.pose.position.z = 1.0891195;
                pose_pub_.publish(pose_msg);
            }

        } else if (triggertest_ && class_id == 1 && conf > 0.7 && depth_m>1.0 && depth_m<3.0 && depth_m<depth_min_) {
            trigger_ = true;
            tri_msg.data = true;
            depth_min_ = depth_m;
            // 计算目标点并发布
            geometry_msgs::PoseStamped pose_msg;
            pose_msg.header.stamp = ros::Time::now();
            pose_msg.header.frame_id = "map";

            // 简化处理，这里直接使用point_world
            // 需要根据python逻辑进行sin、cos位移
            float wh_ = wh_ratio;
            float cos_ = wh_/wh_real_;
            float sin_ = sqrt(1 - cos_*cos_);
            Eigen::Vector3f pointxyz_body = point_body;

            // 按python逻辑移动点
            pointxyz_body[0] = pointxyz_body[0] - sin_ * std::max((depth_m -1),distance_);
            pointxyz_body[2] = pointxyz_body[2] + cos_*std::max(depth_m,distance_);

            Eigen::Vector3f new_world = bodytoworld(pointxyz_body, trans, R_b2w);
            if (depth_m <4) {
                hover_msg.data = true; 
                // 回到预设点
                new_world[0] = -0.9407164;
                new_world[1] = 0.7008502;
                new_world[2] = 1.0891195;
            }

            pose_msg.pose.position.x = new_world[0];
            pose_msg.pose.position.y = new_world[1];
            pose_msg.pose.position.z = new_world[2];

            // 简单估计yaw
            float yaw = atan2(new_world[1]-trans[1], new_world[0]-trans[0]);
            tf2::Quaternion q;
            q.setRPY(0,0,yaw);
            pose_msg.pose.orientation = tf2::toMsg(q);

            pose_pub_.publish(pose_msg);

            // error发布
            error_msg.data = (width/2.0 - cx)/width;
        }
    }

    discover_msg.data = discover && triggertest_;

    // 发布消息
    tri_pub_.publish(tri_msg);
    tuning_pub_.publish(tuning_msg);
    error_pub_.publish(error_msg);
    discover_pub_.publish(discover_msg);
    hover_pub_.publish(hover_msg);

    // 最后发布全部检测
    publish_detections(results);
}

