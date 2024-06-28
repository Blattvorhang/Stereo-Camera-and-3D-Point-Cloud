#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/core.hpp>

/// \brief Represents the Perspective Camera Model.
class Camera
{
public:
    /// \brief Constructs the camera model.
    /// \param K The intrinsic calibration matrix.
    /// \param distortion_coeffs Distortion coefficients on the form [k1, k2, p1, p2, k3].
    explicit Camera(const cv::Mat &K,
                    const cv::Mat &distortion_coeffs);
    Camera();

    /// \return The intrinsic calibration matrix.
    cv::Mat getIntrinsicMatrix() const;

    /// \return The distortion coefficients.
    cv::Mat getDistortionCoeffs() const;

    /// \return The camera projection matrix.
    cv::Mat getProjectionMatrix() const;

    /*
    /// \brief Projects a world point into pixel coordinates.
    /// \param world_point A 3D point in world coordinates.
    Eigen::Vector2d projectWorldPoint(Eigen::Vector3d world_point) const;

    /// \brief Projects a set of world points into pixel coordinates.
    /// \param world_points A set of 3D points in world coordinates.
    Eigen::Matrix2Xd projectWorldPoints(Eigen::Matrix3Xd world_points) const;

    /// \brief Undistorts an image corresponding to the camera model.
    /// \param distorted_image The original, distorted image.
    /// \return The undistorted image.
    cv::Mat undistortImage(cv::Mat distorted_image) const;
    */

private:
    cv::Mat K_;
    cv::Mat distortion_coeffs_;
    cv::Mat projection_matrix_;
};

#endif // CAMERA_H
