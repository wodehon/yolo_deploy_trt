#ifndef YOLOV5_TRT_ROS_H
#define YOLOV5_TRT_ROS_H

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <yolo_deploy_trt/Detection2D.h>
#include <yolo_deploy_trt/Detections2D.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace nvinfer1;

const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

class YOLOv5 {
public:
    YOLOv5(const std::string& engine_name, ros::NodeHandle& nh);
    ~YOLOv5();

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthCallback(const sensor_msgs::ImageConstPtr& msg);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);

private:
    void deserializeEngine(const std::string& engine_name);
    void prepareBuffers();
    void infer(const cv::Mat& image, std::vector<Detection>& results);
    void processFrame(const cv::Mat& color_frame, const cv::Mat& depth_frame);
    Eigen::Vector3f pixelToWorld(float u, float v, float depth);

    void publishDetectedImage(const cv::Mat& img);
    void publishDetections(const std::vector<Detection>& results);
    void publishPoseResult(const Eigen::Vector3f& world_point_);

    // for CUDA and TensorRT
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    cudaStream_t stream_;
    float* gpu_buffers_[2];
    float* cpu_output_buffer_;

    // ROS Subscribers and Publishers
    ros::Subscriber color_sub_;
    ros::Subscriber depth_sub_;
    ros::Subscriber pose_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Publisher detection_pub_;
    ros::Publisher img_res_pub_;
    ros::Publisher pose_pub_;

    // Camera parameters
    Eigen::Matrix3f camera_matrix_;
    // Eigen::Vector4f distortion_coeffs_;
    Eigen::Matrix4f camera_to_body_; // Camera to body transform
    bool camera_info_received_ = false;

    // Depth image and drone pose
    cv::Mat current_depth_;
    Eigen::Matrix4f body_to_world_;  // UAV pose in world frame
    bool pose_received_ = false;

    // Configurations
    const float MAX_Z = 10.0; // Max depth in meters

};

#endif // YOLOV5_TRT_ROS_H
