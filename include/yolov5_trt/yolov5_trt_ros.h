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
#include <std_msgs/String.h>

using namespace nvinfer1;

class YOLOv5 {
public:
    YOLOv5(const std::string& engine_name, ros::NodeHandle& nh);
    ~YOLOv5();

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
    void deserializeEngine(const std::string& engine_name);
    void prepareBuffers();
    void infer(const cv::Mat& image, std::vector<std::vector<Detection>>& results);

    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    cudaStream_t stream_;
    float* gpu_buffers_[2];
    float* cpu_output_buffer_;

    ros::Subscriber image_sub_;
    ros::Publisher detection_pub_;
};

#endif // YOLOV5_TRT_ROS_H
