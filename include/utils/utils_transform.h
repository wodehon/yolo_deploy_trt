#ifndef UTILS_TRANSFORM_H
#define UTILS_TRANSFORM_H

#include <Eigen/Dense>
#include <math.h>
#include <limits>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

Eigen::Vector3f px2xy(float u, float v, const cv::Mat& K, const cv::Mat& D, float z = 1.0f) {
    // 构造像素点
    std::vector<cv::Point2f> pixel_points = {cv::Point2f(u, v)};
    std::vector<cv::Point2f> undistorted_points;

    // 使用 OpenCV 的 undistortPoints 进行畸变校正
    cv::undistortPoints(pixel_points, undistorted_points, K, D);
    // cv::undistortPoints(pixel_points, undistorted_points, K, D, cv::Mat(), K);
    // cv::undistortPoints(pixel_points, undistorted_points, K, D, cv::noArray(), K);

    // 恢复到相机坐标系下
    return Eigen::Vector3f(undistorted_points[0].x * z, undistorted_points[0].y * z, z);
}

inline Eigen::Vector3f camtobody(const Eigen::Vector3f &point_cam) {
    Eigen::Matrix3f cam2body;
    // use ENU(FLU) for px4 
    cam2body << 0.f, 0.f, 1.f,
               -1.f, 0.f, 0.f,
                0.f,-1.f, 0.f;
    Eigen::Vector3f offset(-0.02f, 0.f, 0.f);
    return cam2body * point_cam + offset;
}

inline Eigen::Matrix3f quatToRot(float w, float x, float y, float z){
    Eigen::Quaternionf q(w,x,y,z);
    return q.normalized().toRotationMatrix();
}

inline Eigen::Vector3f bodytoworld(const Eigen::Vector3f &point_body, const Eigen::Vector3f &trans, const Eigen::Matrix3f &R_b2w) {
    Eigen::Vector3f point_world = R_b2w * point_body + trans;
    Eigen::Matrix3f ros2px4;
    ros2px4 << 0.f,-1.f, 0.f,
               1.f, 0.f, 0.f,
               0.f, 0.f, 1.f;
    // point_world = ros2px4 * point_world; // mavros -> px4
    return point_world;
}

inline Eigen::Vector3f truncPoint(const Eigen::Vector3f &point, const Eigen::Vector3f &trans, const Eigen::Vector4f &range) {
    float range_x = range[0];
    float range_y = range[1];
    float floor = range[2];
    float range_z = range[3]; // 2.5f - 0.1f
    float i_ = point[0] - trans[0];
    float j_ = point[1] - trans[1];
    float k_ = point[2] - trans[2];
    std::cout << "[Info ] i_: " << i_ << ",  j_: " << j_ << ",  k_: " << k_ << std::endl;

    float inf_f = std::numeric_limits<float>::infinity();

    float ti = (std::fabs(i_) < 0.01f) ? inf_f : (((i_>0?range_x:-range_x) - trans[0]) / i_);
    float tj = (std::fabs(j_) < 0.01f) ? inf_f : (((j_>0?range_y:-range_y) - trans[1]) / j_);
    float tk = 0.0f;
    if (k_ > 0.f) tk = (range_z - trans[2])/k_;
    else if (k_ <0.f) tk = (0.f - trans[2])/k_;
    else tk = inf_f;
    std::cout << "[Info ] ti: " << ti << ",  tj: " << tj << ",  tk: " << tk << std::endl;
    float t = fminf(ti,fminf(tj,tk));

    float x_ = trans[0] + t*i_;
    float y_ = trans[1] + t*j_;
    float z_ = trans[2] + t*k_;

    return Eigen::Vector3f(x_,y_,z_);
}

// inline Eigen::Vector3f truncPoint(const Eigen::Vector3f &point, const Eigen::Vector3f &trans, const Eigen::Vector4f &range) {
//     const float range_x = range[0];
//     const float range_y = range[1];
//     const float range_z_min = range[2];
//     const float range_z_max = range[3];

//     // 射线方向向量
//     Eigen::Vector3f dir = point - trans;

//     // 用于存储 t 值的数组
//     float t_min = std::numeric_limits<float>::infinity();

//     // 初始化交点坐标
//     Eigen::Vector3f intersection = trans;

//     // 检查 x 方向
//     if (std::abs(dir[0]) > 1e-6) { // 避免除以 0
//         float t = (dir[0] > 0 ? range_x - trans[0] : -range_x - trans[0]) / dir[0];
//         if (t > 0) { // t 必须为正
//             float y = trans[1] + t * dir[1];
//             float z = trans[2] + t * dir[2];
//             if (std::abs(y) <= range_y && z >= range_z_min && z <= range_z_max) {
//                 t_min = t;
//                 intersection = Eigen::Vector3f(trans[0] + t * dir[0], y, z);
//             }
//         }
//     }

//     // 检查 y 方向
//     if (std::abs(dir[1]) > 1e-6) {
//         float t = (dir[1] > 0 ? range_y - trans[1] : -range_y - trans[1]) / dir[1];
//         if (t > 0) {
//             float x = trans[0] + t * dir[0];
//             float z = trans[2] + t * dir[2];
//             if (std::abs(x) <= range_x && z >= range_z_min && z <= range_z_max && t < t_min) {
//                 t_min = t;
//                 intersection = Eigen::Vector3f(x, trans[1] + t * dir[1], z);
//             }
//         }
//     }

//     // 检查 z 方向
//     if (std::abs(dir[2]) > 1e-6) {
//         float t = (dir[2] > 0 ? range_z_max - trans[2] : range_z_min - trans[2]) / dir[2];
//         if (t > 0) {
//             float x = trans[0] + t * dir[0];
//             float y = trans[1] + t * dir[1];
//             if (std::abs(x) <= range_x && std::abs(y) <= range_y && t < t_min) {
//                 t_min = t;
//                 intersection = Eigen::Vector3f(x, y, trans[2] + t * dir[2]);
//             }
//         }
//     }

//     // 返回交点
//     return intersection;
// }

#endif // UTILS_TRANSFORM_H