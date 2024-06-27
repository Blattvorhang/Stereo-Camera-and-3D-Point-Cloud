#include "../include/camera.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

PerspectiveCameraModel::PerspectiveCameraModel(const Eigen::Matrix3d &K,
                                               const Sophus::SE3d &pose_world_camera,
                                               const Vector5d &distortion_coeffs)
    : K_{K}, pose_world_camera_{pose_world_camera}, distortion_coeffs_{distortion_coeffs}
{
    camera_projection_matrix_ = computeCameraProjectionMatrix();
}

Sophus::SE3d PerspectiveCameraModel::getPose() const
{
    return pose_world_camera_;
}

Eigen::Matrix3d PerspectiveCameraModel::getCalibrationMatrix() const
{
    return K_;
}

PerspectiveCameraModel::Matrix34d PerspectiveCameraModel::getCameraProjectionMatrix() const
{
    return camera_projection_matrix_;
}

Eigen::Vector2d PerspectiveCameraModel::projectWorldPoint(Eigen::Vector3d world_point) const
{
    // DONE: Implement projection using camera_projection_matrix_.
    return (camera_projection_matrix_ * world_point.homogeneous()).hnormalized();
}

Eigen::Matrix2Xd PerspectiveCameraModel::projectWorldPoints(Eigen::Matrix3Xd world_points) const
{
    // DONE: Optionally implement projection using camera_projection_matrix_.
    return (camera_projection_matrix_ * world_points.colwise().homogeneous()).colwise().hnormalized();
}

PerspectiveCameraModel::Matrix34d PerspectiveCameraModel::computeCameraProjectionMatrix()
{
    // DONE: Compute camera projection matrix.
    return K_ * pose_world_camera_.inverse().matrix3x4();
}

cv::Mat PerspectiveCameraModel::undistortImage(cv::Mat distorted_image) const
{
    // Convert to cv::Mats
    cv::Mat K_cv;
    cv::eigen2cv(K_, K_cv);
    cv::Mat dist_coeffs_cv;
    cv::eigen2cv(distortion_coeffs_, dist_coeffs_cv);

    // Undistort image.
    cv::Mat undistorted_image;
    cv::undistort(distorted_image, undistorted_image, K_cv, dist_coeffs_cv);

    return undistorted_image;
}
