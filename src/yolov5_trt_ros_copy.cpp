#include "yolov5_trt_ros.h"

static Logger gLogger;

YOLOv5::YOLOv5(const std::string& engine_name, ros::NodeHandle& nh)
    : runtime_(nullptr), engine_(nullptr), context_(nullptr), cpu_output_buffer_(nullptr) {
    // Deserialize engine
    deserializeEngine(engine_name);
    std::cout<< "[INFO] engine deserial successed!" << std::endl;

    // Prepare CUDA buffers
    prepareBuffers();

    // Initialize camera matrix and transforms
    camera_matrix_.setIdentity();
    camera_to_body_.setIdentity();
    body_to_world_.setIdentity();

    // Get topic name from parameter or use default value
    std::string color_topic;
    nh.param<std::string>("color_topic", color_topic, "/camera/color/image_raw");
    std::string depth_topic;
    nh.param<std::string>("depth_topic", depth_topic, "/camera/aligned_depth_to_color/image_raw");
    std::string pose_topic;
    nh.param<std::string>("pose_topic", pose_topic, "/mavros/local_position/pose");
    std::string detect_topic;
    nh.param<std::string>("detect_topic", detect_topic, "/yolov5/detections");
    std::string detimg_topic;
    nh.param<std::string>("detimg_topic", detimg_topic, "/yolov5/detImg");
    std::string res_topic;
    nh.param<std::string>("res_topic", res_topic, "/yolov5/pose");

    // Initialize ROS subscribers and publishers
    color_sub_ = nh.subscribe(color_topic, 1, &YOLOv5::imageCallback, this);
    depth_sub_ = nh.subscribe(depth_topic, 1, &YOLOv5::depthCallback, this);
    pose_sub_ = nh.subscribe(pose_topic, 1, &YOLOv5::poseCallback, this);
    camera_info_sub_ = nh.subscribe("/camera/color/camera_info", 1, &YOLOv5::cameraInfoCallback, this);

    detection_pub_ = nh.advertise<yolo_deploy_trt::Detections2D>(detect_topic, 1);
    img_res_pub_ = nh.advertise<sensor_msgs::Image>(detimg_topic, 1);
    pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>(res_topic, 1);
    std::cout<< "[INFO] YOLOV5 init successed!" << std::endl;
}

YOLOv5::~YOLOv5() {
    cudaStreamDestroy(stream_);
    CUDA_CHECK(cudaFree(gpu_buffers_[0]));
    CUDA_CHECK(cudaFree(gpu_buffers_[1]));
    delete[] cpu_output_buffer_;

    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}

void YOLOv5::deserializeEngine(const std::string& engine_name) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to read engine file");
    }

    size_t size;
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
    CUDA_CHECK(cudaMalloc((void**)&gpu_buffers_[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&gpu_buffers_[1], kBatchSize * kOutputSize * sizeof(float)));
    cpu_output_buffer_ = new float[kBatchSize * kOutputSize];
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

void YOLOv5::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    if (!camera_info_received_ || current_depth_.empty()) {
        ROS_WARN("Camera info or depth frame not received yet");
        return;
    }

    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    std::cout<<"[INFO] current_image_received_"<<std::endl;
    processFrame(cv_ptr->image, current_depth_);
    std::cout<<"[INFO] processFrame_finish_"<<std::endl;
}

void YOLOv5::depthCallback(const sensor_msgs::ImageConstPtr& msg) {
    current_depth_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    std::cout<<"[INFO] current_depth_received_"<<std::endl;
}
void YOLOv5::poseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
    Eigen::Quaternionf q(msg->pose.orientation.w, msg->pose.orientation.x, 
                         msg->pose.orientation.y, msg->pose.orientation.z);
    Eigen::Matrix3f rotation = q.toRotationMatrix();

    body_to_world_.setIdentity();
    body_to_world_.block<3, 3>(0, 0) = rotation;
    body_to_world_.block<3, 1>(0, 3) = Eigen::Vector3f(msg->pose.position.x, 
                                                       msg->pose.position.y, 
                                                       msg->pose.position.z);
    
    pose_received_ = true;
}

// void YOLOv5::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
//     float camera_matrix_float[9];
//     for (int i = 0; i < 9; ++i) {
//         camera_matrix_float[i] = static_cast<float>(msg->K[i]);
//     }
//     camera_matrix_ = Eigen::Map<Eigen::Matrix3f>(camera_matrix_float);
//     camera_info_received_ = true;
// }

void YOLOv5::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
    camera_matrix_ << msg->K[0], msg->K[1], msg->K[2],
                      msg->K[3], msg->K[4], msg->K[5],
                      msg->K[6], msg->K[7], msg->K[8];
    camera_info_received_ = true;
    std::cout<<"[INFO] camera_info_received_"<<std::endl;
}

void YOLOv5::infer(const cv::Mat& image, std::vector<Detection>& results) {
    std::vector<cv::Mat> img_batch = {image};
    cuda_batch_preprocess(img_batch, gpu_buffers_[0], kInputW, kInputH, stream_);
    // cuda_preprocess(const_cast<uint8_t*>(image.ptr()), image.cols, image.rows, gpu_buffers_[0], kInputW, kInputH, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    context_->enqueueV2((void**)gpu_buffers_, stream_, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer_, gpu_buffers_[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);
    nms(results, cpu_output_buffer_, kConfThresh, kNmsThresh);
}

void YOLOv5::processFrame(const cv::Mat& color_frame, const cv::Mat& depth_frame) {
    std::cout<<"[INFO] processFrame_start_"<<std::endl;
    std::vector<Detection> detections;
    infer(color_frame, detections);
    Eigen::Vector3f world_point;

    cv::Mat color_img = color_frame.clone();
    draw_bbox(color_img, detections);
    std::cout<<"[INFO] draw_bbox_finish_"<<std::endl;
    publishDetectedImage(color_frame);
    std::cout<<"[INFO] publishDetectedImage_finish_"<<std::endl;
    publishDetections(detections);
    std::cout<<"[INFO] publishDetections_finish_"<<std::endl;
    if (!detections.empty() && pose_received_) {
        for (const auto& det : detections) {
            float u = det.bbox[0];
            float v = det.bbox[1];
            float depth = depth_frame.at<uint16_t>(v, u) * 0.001f;

            if (depth == 0) {
                depth = MAX_Z;
            }

            world_point = pixelToWorld(u, v, depth);
            ROS_INFO("Detected object at world coordinates: x=%f, y=%f, z=%f", world_point.x(), world_point.y(), world_point.z());
        }
        publishPoseResult(world_point);
    }
    std::cout<<"[INFO] processFrame_end_"<<std::endl;
}

Eigen::Vector3f YOLOv5::pixelToWorld(float u, float v, float depth) {
    // Step 1: pixel -> cam
    Eigen::Vector3f pixel_point(u, v, 1.0f);
    Eigen::Vector3f norm_cam = camera_matrix_.inverse() * pixel_point * depth;

    // Step 2: cam -> body
    Eigen::Vector4f cam_point_homo(norm_cam.x(), norm_cam.y(), norm_cam.z(), 1.0f);  // 齐次相机坐标
    Eigen::Vector4f body_point_homo = camera_to_body_ * cam_point_homo;

    // Step 3: body -> world
    Eigen::Vector4f world_point_homo = body_to_world_ * body_point_homo;

    // 返回三维世界坐标
    return Eigen::Vector3f(world_point_homo.x(), world_point_homo.y(), world_point_homo.z());
}

void YOLOv5::publishDetectedImage(const cv::Mat& img) {
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    cv_bridge::CvImage cv_img(header, sensor_msgs::image_encodings::BGR8, img);
    img_res_pub_.publish(cv_img.toImageMsg());
}

void YOLOv5::publishDetections(const std::vector<Detection>& results) {
    yolo_deploy_trt::Detections2D msg;
    msg.header.stamp = ros::Time::now();

    for (const auto& det : results) {
        yolo_deploy_trt::Detection2D detection;
        detection.confidence = det.conf;
        detection.class_id = det.class_id;

        detection.box_min.x = det.bbox[0] - 0.5 * det.bbox[2];
        detection.box_min.y = det.bbox[1] - 0.5 * det.bbox[3];
        detection.box_max.x = det.bbox[0] + 0.5 * det.bbox[2];
        detection.box_max.y = det.bbox[1] + 0.5 * det.bbox[3];

        msg.detections.push_back(detection);
    }

    detection_pub_.publish(msg);
}

void YOLOv5::publishPoseResult(const Eigen::Vector3f& world_point_) {
    geometry_msgs::PoseStamped res_pose;
    res_pose.header.stamp = ros::Time::now();
    res_pose.pose.position.x = world_point_(0);
    res_pose.pose.position.y = world_point_(1);
    res_pose.pose.position.z = world_point_(2);
    pose_pub_.publish(res_pose);
}
