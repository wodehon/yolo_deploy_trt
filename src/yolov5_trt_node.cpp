#include "ros/ros.h"
#include "yolov5_trt/yolov5_trt_ros.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "yolov5_trt_node");
    ros::NodeHandle nh("~");

    std::string engine_file;
    nh.param<std::string>("engine_file", engine_file, "yolov5.engine");

    try {
        YOLOv5 yolov5(engine_file, nh);
        ros::spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to initialize YOLOv5: %s", e.what());
        return -1;
    }

    return 0;
}
