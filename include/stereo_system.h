#ifndef STEREO_SYSTEM_H
#define STEREO_SYSTEM_H

#include <string>
#include <opencv2/opencv.hpp>
#include "../include/camera.h"

/// \brief Whole stereo system class.
class StereoSystem
{
public:
    /// \brief Constructs a stereo system object.
    /// \param param_path Optional path to parameter files.
    /// \param camera_id The camera ID to use, -1 for file input.
    /// \param single_camera_width The width of the single camera image.
    /// \param single_camera_height The height of the single camera image.
    /// \param enable_debug Shows additional debug/educational figures and prints if true.
    explicit StereoSystem(const std::string &param_path = "../calibration/",
                          int camera_id = 0,
                          int single_camera_width = 1280,
                          int single_camera_height = 720,
                          bool enable_debug = false);

    /// \brief Runs the stereo system.
    void run();

    void calibrateStereoCameras();

    void rectifyImages(const cv::Mat &left_image, const cv::Mat &right_image,
                       cv::Mat &rectified_left_image, cv::Mat &rectified_right_image);

    void computeDisparity(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity);
    void computeDepthMap(const cv::Mat &disparity, cv::Mat &depth_map);

private:
    /// @brief Checks if the matrix has the expected size.
    /// @param mat
    /// @param expected_rows 
    /// @param expected_cols 
    void checkSize(const cv::Mat& mat, int expected_rows, int expected_cols);

    void readCalibrationParameters(const std::string &param_path);
    void captureImages(cv::VideoCapture &cap, cv::Mat &left_image, cv::Mat &right_image);

    int camera_id_;
    int width_;
    int height_;
    bool enable_debug_;
    Camera left_camera_;
    Camera right_camera_;
    cv::Mat R_;
    cv::Mat T_;
    cv::Mat R1_, R2_, P1_, P2_, Q_;
};

#endif // STEREO_SYSTEM_H
