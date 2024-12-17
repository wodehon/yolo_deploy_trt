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
    // cv::undistortPoints(pixel_points, undistorted_points, K, D);
    // cv::undistortPoints(pixel_points, undistorted_points, K, D, cv::Mat(), K);
    cv::undistortPoints(pixel_points, undistorted_points, K, D, cv::noArray(), K);

    // 恢复到相机坐标系下
    return Eigen::Vector3f(undistorted_points[0].x * z, undistorted_points[0].y * z, z);
}

inline Eigen::Vector3f camtobody(const Eigen::Vector3f &point_cam) {
    Eigen::Matrix3f cam2body;
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
    return point_world;
}

inline Eigen::Vector3f truncPoint(const Eigen::Vector3f &point, const Eigen::Vector3f &trans) {
    float range_x = 2.5f;
    float range_y = 2.5f;
    float range_z = 2.4f; // 2.5f - 0.1f
    float i_ = point[0] - trans[0];
    float j_ = point[1] - trans[1];
    float k_ = point[2] - trans[2];

    float inf_f = std::numeric_limits<float>::infinity();

    float ti = (i_ == 0.f) ? inf_f : (((i_>0?range_x:-range_x) - trans[0]) / i_);
    float tj = (j_ == 0.f) ? inf_f : (((j_>0?range_y:-range_y) - trans[1]) / j_);
    float tk;
    if (k_ > 0.f) tk = (range_z - trans[2])/k_;
    else if (k_ <0.f) tk = (0.f - trans[2])/k_;
    else tk = inf_f;

    float t = fminf(ti,fminf(tj,tk));

    float x_ = trans[0] + t*i_;
    float y_ = trans[1] + t*j_;
    float z_ = trans[2] + t*k_;

    return Eigen::Vector3f(x_,y_,z_);
}

#endif // UTILS_TRANSFORM_H