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
#include <geometry_msgs/Point.h>
#include <yolo_deploy_trt/Detection2D.h>
#include <yolo_deploy_trt/Detections2D.h>

using namespace nvinfer1;

class YOLOv5 {
public:
    YOLOv5(const std::string& engine_name, ros::NodeHandle& nh);
    ~YOLOv5();

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
    void deserializeEngine(const std::string& engine_name);
    void prepareBuffers();
    void publish_detected_image(cv::Mat& img);
    void publish_detections(const std::vector<Detection>& results);
    void infer(const cv::Mat& image, std::vector<Detection>& results);
    
    // for cuda and tensorrt
    IRuntime* runtime_;
    ICudaEngine* engine_;
    IExecutionContext* context_;
    cudaStream_t stream_;
    float* gpu_buffers_[2];
    float* cpu_output_buffer_;

    // ros Subscriber and Publisher
    ros::Subscriber image_sub_;
    ros::Publisher detection_pub_;
    ros::Publisher img_res_pub_;

    // filter
    // static std::array<float, 4> previous_;
    // static std::array<float, 4> velocity_;
    // Detection filter_;
    Detection filter_;
    // OneEuroFilter
    // double frequency = 120 ; // Hz
    // double mincutoff = 1.0 ; // Hz
    // double beta = 0.1 ;      
    // double dcutoff = 1.0 ; 
    // OneEuroFilter f_;
};

#endif // YOLOV5_TRT_ROS_H

// from utils/type.h
// struct alignas(float) Detection {
//   float bbox[4];  // center_x center_y w h
//   float conf;  // bbox_conf * cls_conf
//   float class_id;
//   float mask[32];
// };