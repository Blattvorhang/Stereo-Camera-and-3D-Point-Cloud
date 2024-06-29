#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "../include/stereo_system.h"
#include "../include/disparity.h"

StereoSystem::StereoSystem(const std::string &param_path,
                           int camera_id,
                           int single_camera_width,
                           int single_camera_height,
                           bool enable_debug)
    : camera_id_{camera_id},
      width_{single_camera_width},
      height_{single_camera_height},
      enable_debug_{enable_debug}
{
    readCalibrationParameters(param_path);
}

void StereoSystem::checkSize(const cv::Mat &mat, int expected_rows, int expected_cols)
{
    if (mat.rows != expected_rows || mat.cols != expected_cols)
    {
        throw std::runtime_error("Matrix size does not match the expected size.");
    }
}

void StereoSystem::readCalibrationParameters(const std::string &param_path)
{
    std::stringstream filepath;
    filepath << param_path << "calib_param.yml";
    cv::FileStorage fs(filepath.str(), cv::FileStorage::READ);

    cv::Mat cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR;

    // Read the camera matrices and distortion coefficients.
    fs["cameraMatrixL"] >> cameraMatrixL;
    fs["distCoeffsL"] >> distCoeffsL;
    fs["cameraMatrixR"] >> cameraMatrixR;
    fs["distCoeffsR"] >> distCoeffsR;

    // Read the rotation and translation matrices.
    fs["R"] >> R_;
    fs["T"] >> T_;

    // Check the sizes of the matrices.
    try
    {
        checkSize(cameraMatrixL, 3, 3); // Assuming camera matrices are 3x3
        checkSize(distCoeffsL, 1, 5);   // Assuming distortion coefficients are 1x5
        checkSize(cameraMatrixR, 3, 3);
        checkSize(distCoeffsR, 1, 5);
        checkSize(R_, 3, 3); // Rotation matrix is 3x3
        checkSize(T_, 3, 1); // Translation vector is 3x1
    }
    catch (const std::runtime_error &e)
    {
        throw std::runtime_error("Error reading calibration parameters: " + std::string(e.what()));
    }

    std::cout << "Camera matrices and distortion coefficients loaded successfully." << std::endl;

    // Store the matrices and vectors.
    left_camera_ = Camera{cameraMatrixL, distCoeffsL};
    right_camera_ = Camera{cameraMatrixR, distCoeffsR};

    fs.release();
}

void StereoSystem::calibrateStereoCameras()
{
    cv::stereoRectify(left_camera_.getIntrinsicMatrix(), left_camera_.getDistortionCoeffs(),
                      right_camera_.getIntrinsicMatrix(), right_camera_.getDistortionCoeffs(),
                      cv::Size(width_, height_),
                      R_, T_, R1_, R2_, P1_, P2_, Q_,
                      cv::CALIB_ZERO_DISPARITY, 0, cv::Size(width_, height_));
    if (enable_debug_)
    {
        std::cout << "R1: " << std::endl << R1_ << std::endl;
        std::cout << "R2: " << std::endl << R2_ << std::endl;
        std::cout << "P1: " << std::endl << P1_ << std::endl;
        std::cout << "P2: " << std::endl << P2_ << std::endl;
        std::cout << "Q: " << std::endl << Q_ << std::endl;
    }
}

void StereoSystem::rectifyImages(const cv::Mat &ori_left, const cv::Mat &ori_right,
                                 cv::Mat &rectified_ori_left, cv::Mat &rectified_ori_right)
{
    cv::Mat rect_map_L1, rect_map_L2, rect_map_R1, rect_map_R2;

    cv::initUndistortRectifyMap(left_camera_.getIntrinsicMatrix(), left_camera_.getDistortionCoeffs(),
                                R1_, P1_, cv::Size(width_, height_),
                                CV_32FC1, rect_map_L1, rect_map_L2);
    cv::initUndistortRectifyMap(right_camera_.getIntrinsicMatrix(), right_camera_.getDistortionCoeffs(),
                                R2_, P2_, cv::Size(width_, height_),
                                CV_32FC1, rect_map_R1, rect_map_R2);

    cv::remap(ori_left, rectified_ori_left, rect_map_L1, rect_map_L2, cv::INTER_LINEAR);
    cv::remap(ori_right, rectified_ori_right, rect_map_R1, rect_map_R2, cv::INTER_LINEAR);
}

void StereoSystem::captureImages(cv::VideoCapture &cap, cv::Mat &left_image, cv::Mat &right_image)
{
    static cv::Mat frame;
    cap >> frame;
    left_image = frame.colRange(0, frame.cols / 2);
    right_image = frame.colRange(frame.cols / 2, frame.cols);
}

void StereoSystem::computeDisparity(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &disparity)
{
    DisparityMapGenerator disparity_map_generator(left_image, right_image, DisparityMapGenerator::SGBM);
    disparity_map_generator.computeDisparity(disparity);
    if (enable_debug_) {
        disparity_map_generator.displayDisparity();
    }
}

void StereoSystem::computeDepthMap(const cv::Mat &disparity, cv::Mat &depth_map)
{
    
    cv::Mat depth_map_3d;
    cv::reprojectImageTo3D(disparity, depth_map_3d, Q_, true);

    // Extract the Z coordinate from the 3D image.
    std::vector<cv::Mat> xyz;
    cv::split(depth_map_3d, xyz);
    depth_map = xyz[2];
    
    // Alternative method to calculate the depth map.
    // Formula: Z = f * T / disp
    // float baseline = 1 / Q_.at<double>(3, 2);
    // float focal_length = Q_.at<double>(2, 3);
    // depth_map = focal_length * baseline / disparity;

    // Eliminate invalid depth values.
    depth_map.setTo(0, disparity < 0);
}

void StereoSystem::run()
{
    // Open video stream from camera.
    cv::VideoCapture cap;
    if (camera_id_ >= 0)
    {
        cap = cv::VideoCapture(camera_id_);
        if (!cap.isOpened())
        {
            throw std::runtime_error("Could not open camera.");
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width_ * 2);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    }

    calibrateStereoCameras();

    cv::Mat ori_left, ori_right;   // original images
    cv::Mat rect_left, rect_right; // stereo rectified images
    cv::Mat disparity_map;         // disparity map
    cv::Mat depth_map;             // depth map
    cv::Mat depth_map_8u;          // 8-bit depth map
    cv::Mat depth_map_normalized;  // normalized depth map

    while (true)
    {
        if (camera_id_ >= 0) {
            // Capture images from camera.
            captureImages(cap, ori_left, ori_right);
            rectifyImages(ori_left, ori_right, rect_left, rect_right);
        } else { 
            // Load images from file.
            rect_left = cv::imread("../test_imgs/rectified_left.png");
            rect_right = cv::imread("../test_imgs/rectified_right.png");
        }

        // Show rectified images.
        cv::imshow("Rectified Left", rect_left);
        cv::imshow("Rectified Right", rect_right);

        // Save rectified images.
        // cv::imwrite("../test_imgs/rectified_left.png", rect_left);
        // cv::imwrite("../test_imgs/rectified_right.png", rect_right);

        computeDisparity(rect_left, rect_right, disparity_map);

        computeDepthMap(disparity_map, depth_map);

        double max_depth = 1600; // Limit the maximum depth value (unit: mm)
        
        // convert the depth map to 8-bit for visualization
        depth_map.convertTo(depth_map_8u, CV_8U, 255.0 / max_depth);
        // cv::normalize(depth_map_8u, depth_map_normalized, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
        cv::imshow("Depth Map", depth_map_8u);
        
        // Show additional debug/educational figures.
        if (enable_debug_)
        {
            // if (!frame.empty())      { cv::imshow("Original", frame); }
            if (!ori_left.empty())
            {
                cv::imshow("Original Left", ori_left);
            }
            if (!ori_right.empty())
            {
                cv::imshow("Original Right", ori_right);
            }
        }

        if (cv::waitKey(1) == 27) // Break on ESC
        {
            break;
        }
    }
}
