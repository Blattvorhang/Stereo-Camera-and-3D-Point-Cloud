#ifndef STEREO_SYSTEM_H
#define STEREO_SYSTEM_H

#include <string>

/// \brief Whole stereo system class.
class StereoSystem
{
public:
    /// \brief Constructs a stereo system object.
    /// \param data_path Optional path to parameter files.
    explicit StereoSystem(const std::string &data_path = "../calibration/");

    /// \brief Runs the stereo system.
    void run();

private:
    std::string data_path_;
    std::string window_name_;
    int camera_id_;
    int single_camera_width_;
    int single_camera_height_;
    cv::Mat cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR;
    cv::Mat R, T;
};

#endif // STEREO_SYSTEM_H
