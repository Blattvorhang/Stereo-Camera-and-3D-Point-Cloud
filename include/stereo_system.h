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
    /// \param do_visualize Shows additional debug/educational figures when true.
    explicit StereoSystem(const std::string &param_path = "../calibration/",
                          int camera_id = 0,
                          int single_camera_width = 1280,
                          int single_camera_height = 720,
                          bool do_visualize = false);

    /// \brief Runs the stereo system.
    void run();

private:
    /// @brief Checks if the matrix has the expected size.
    /// @param mat
    /// @param expected_rows 
    /// @param expected_cols 
    void checkSize(const cv::Mat& mat, int expected_rows, int expected_cols);

    void readCalibrationParameters(const std::string &param_path);

    int camera_id_;
    int width_;
    int height_;
    bool do_visualize_;
    Camera left_camera_;
    Camera right_camera_;
    cv::Mat R_;
    cv::Mat T_;
};

#endif // STEREO_SYSTEM_H
