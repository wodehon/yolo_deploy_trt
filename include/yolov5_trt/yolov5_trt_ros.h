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
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Float32.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/Detection2DArray.h>
#include <image_transport/image_transport.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace nvinfer1;

// struct Detection {
//     float bbox[4];  // center_x center_y w h
//     float conf;     // bbox_conf * cls_conf
//     float class_id;
//     float mask[32];
// };

class YOLOv5 {
public:
    YOLOv5(const std::string& engine_name, ros::NodeHandle& nh);
    ~YOLOv5();

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthCallback(const sensor_msgs::ImageConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
    void poseCallback(const geometry_msgs::PoseStampedConstPtr &msg);

private:
    void deserializeEngine(const std::string& engine_name);
    void prepareBuffers();
    void publish_detected_image(cv::Mat& img);
    void publish_detections(const std::vector<Detection>& results);

    // 新增的函数，与Python功能对应
    void handleDetections(cv::Mat& frame, const std::vector<Detection>& results);

    void infer(const cv::Mat& image, std::vector<Detection>& results);

    // for cuda and tensorrt
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    cudaStream_t stream_;
    float* gpu_buffers_[2];
    float* cpu_output_buffer_;

    // ROS Subscriber and Publisher
    ros::Subscriber image_sub_;
    ros::Subscriber depth_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Subscriber pose_sub_;

    ros::Publisher detection_pub_;
    ros::Publisher img_res_pub_;

    // 新增的Publisher(根据python代码需求)
    ros::Publisher pose_pub_;
    ros::Publisher tri_pub_;
    ros::Publisher tuning_pub_;
    ros::Publisher error_pub_;
    ros::Publisher discover_pub_;
    ros::Publisher hover_pub_;

    // 保存必要信息
    cv::Mat depth_img_;
    bool depth_received_ = false;
    bool pose_received_ = false;
    bool camera_info_received_ = false;

    cv::Mat K_;
    cv::Mat D_;
    geometry_msgs::PoseStamped current_pose_;

    // 状态变量(根据python代码逻辑)
    bool trigger_ = false;
    bool triggertest_ = false;
    float depth_min_ = 10000.0;
    int count_ = 0;

    // 一些逻辑参数(需根据python逻辑调整)
    float scale_ = 0.001;
    float MAX_Z_ = 10000;
    float wh_real_ = 4.0;
    float distance_ = 1.0;

    // 从参数服务器获取话题名称和Engine文件
    std::string engine_file_;
    std::string image_topic_;

    // YOLO相关常量
    // 这些应在utils.h或model.h中定义。例如：
    // #define kBatchSize 1
    // #define kInputW 640
    // #define kInputH 640
    // #define kConfThresh 0.5f
    // #define kNmsThresh 0.45f
    // #define kMaxNumOutputBbox 1000
    // const char* kInputTensorName = "images";
    // const char* kOutputTensorName = "output";

};

#endif // YOLOV5_TRT_ROS_H