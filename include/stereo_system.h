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
    /// \param enable_debug Shows additional debug/educational figures and prints if true.
    explicit StereoSystem(const std::string &param_path = "../calibration/",
                          int camera_id = 1,
                          int single_camera_width = 1280,
                          int single_camera_height = 720,
                          bool enable_debug = false);

    /// \brief Runs the stereo system.
    void run();

    void calibrateStereoCameras();

    void rectifyImages(const cv::Mat &left_image, const cv::Mat &right_image,
                       cv::Mat &rectified_left_image, cv::Mat &rectified_right_image);

private:
    /// @brief Checks if the matrix has the expected size.
    /// @param mat
    /// @param expected_rows 
    /// @param expected_cols 
    void checkSize(const cv::Mat& mat, int expected_rows, int expected_cols);

    void readCalibrationParameters(const std::string &param_path);

    cv::VideoCapture openCamera(int camera_id, int width, int height);

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
