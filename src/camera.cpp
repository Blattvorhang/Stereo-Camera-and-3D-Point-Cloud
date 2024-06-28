#include "../include/camera.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

Camera::Camera(const cv::Mat &K,
               const cv::Mat &distortion_coeffs)
    : K_{K}, distortion_coeffs_{distortion_coeffs}
{ }

cv::Mat Camera::getIntrinsicMatrix() const
{
    return K_;
}

cv::Mat Camera::getDistortionCoeffs() const
{
    return distortion_coeffs_;
}

cv::Mat Camera::getProjectionMatrix() const
{
    return projection_matrix_;
}

/*
Eigen::Vector2d Camera::projectWorldPoint(Eigen::Vector3d world_point) const
{
    // DONE: Implement projection using camera_projection_matrix_.
    return (camera_projection_matrix_ * world_point.homogeneous()).hnormalized();
}

Eigen::Matrix2Xd Camera::projectWorldPoints(Eigen::Matrix3Xd world_points) const
{
    // DONE: Optionally implement projection using camera_projection_matrix_.
    return (camera_projection_matrix_ * world_points.colwise().homogeneous()).colwise().hnormalized();
}

cv::Mat Camera::undistortImage(cv::Mat distorted_image) const
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
*/
