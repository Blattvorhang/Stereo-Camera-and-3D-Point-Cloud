#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <chrono>
#include <thread>
#include <iostream>
#include "../include/stereo_system.h"
#include "../include/disparity.h"

StereoSystem::StereoSystem(const std::string &param_path,
                           int camera_id,
                           int single_camera_width,
                           int single_camera_height,
                           DisparityMapGenerator::DisparityMethod method,
                           bool enable_debug)
	:
      camera_id_{camera_id},
      width_{single_camera_width},
      height_{single_camera_height},
      method{ method },
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

void StereoSystem::createPointCloud(const cv::Mat& _3dImage, const cv::Mat& colorImage,
                                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pointCloud)
{
    pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    pointCloud->points.resize(_3dImage.total());

    for (int i = 0; i < _3dImage.rows; ++i)
    {
        for (int j = 0; j < _3dImage.cols; ++j)
        {
            const cv::Vec3f& point = _3dImage.at<cv::Vec3f>(i, j);
            
            if (point[2] > 0)
            {
                pointCloud->points[i * _3dImage.cols + j].x = point[0];
                pointCloud->points[i * _3dImage.cols + j].y = point[1];
                pointCloud->points[i * _3dImage.cols + j].z = point[2];
                pointCloud->points[i * _3dImage.cols + j].r = colorImage.at<cv::Vec3b>(i, j)[2];
                pointCloud->points[i * _3dImage.cols + j].g = colorImage.at<cv::Vec3b>(i, j)[1];
                pointCloud->points[i * _3dImage.cols + j].b = colorImage.at<cv::Vec3b>(i, j)[0];
            }
        }
    }

    pointCloud->width = _3dImage.cols;
    pointCloud->height = _3dImage.rows;
    pointCloud->is_dense = false;
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
            // rect_left = cv::imread("../test_imgs/im2.png");
            // rect_right = cv::imread("../test_imgs/im6.png");
            rect_left = cv::imread("../test_imgs/rectified_left.png");
            rect_right = cv::imread("../test_imgs/rectified_right.png");
        }

        // Show rectified images.
        cv::imshow("Rectified Left", rect_left);
        cv::imshow("Rectified Right", rect_right);

        // Save rectified images.
        // cv::imwrite("../test_imgs/rectified_left.png", rect_left);
        // cv::imwrite("../test_imgs/rectified_right.png", rect_right);

        DisparityMapGenerator disparity_map_generator(rect_left, rect_right, this->method);
        // 开始计时
        int64 start = cv::getTickCount();
        disparity_map_generator.computeDisparity(disparity_map);
        // 结束计时
        int64 end = cv::getTickCount();
        double duration = (end - start) / cv::getTickFrequency(); // 计算BM算法的运行时间（秒）
		std::cout << "算法的运行时间为：" << duration << "秒" << std::endl;
        disparity_map_generator.displayDisparity();

        computeDepthMap(disparity_map, depth_map);

        double max_depth = 1200; // Limit the maximum depth value (unit: mm)
        
        // convert the depth map to 8-bit for visualization
        depth_map.convertTo(depth_map_8u, CV_8U, 255.0 / max_depth);
        // cv::normalize(depth_map_8u, depth_map_normalized, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
        cv::imshow("Depth Map", depth_map_8u);

        cv::Mat _3dImage;
        cv::Mat colorImage;

        cv::resize(rect_left, colorImage, cv::Size(), 0.5, 0.5);
        cv::reprojectImageTo3D(disparity_map, _3dImage, Q_, true);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        createPointCloud(_3dImage, colorImage, pointCloud);

        if (pointCloud->points.size() > 0)
        {
            std::cout << "Point cloud size: " << pointCloud->points.size() << std::endl;
        }
        else
        {
            std::cout << "Point cloud is empty." << std::endl;
        }

        // Show the point cloud.
        /*pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(204 / 255.0, 204 / 255.0, 204 / 255.0);  // Light gray
        viewer->addPointCloud<pcl::PointXYZRGB>(pointCloud, "point cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud");
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();
        viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);*/
        
        // Show additional debug/educational figures.
        if (enable_debug_)
        {
            // if (!frame.empty())      { cv::imshow("Original", frame); }
            if (!ori_left.empty() && !ori_right.empty())
            {
                // Combine the original images side by side.
                cv::Mat ori_combined;
                cv::hconcat(ori_left, ori_right, ori_combined);

                // Draw horizontal lines on the original images. 
                for (int i = 50; i < ori_combined.rows - 1; i += 50)
                {
                    cv::line(ori_combined, cv::Point(0, i), cv::Point(ori_combined.cols, i), cv::Scalar(0, 0, 255), 1);
                }

                cv::imshow("Horizontal Lines (Original)", ori_combined);
            }
            if (!rect_left.empty() && !rect_right.empty())
            {
                // Combine the rectified images side by side.
                cv::Mat rect_combined;
                cv::hconcat(rect_left, rect_right, rect_combined);

                // Draw horizontal lines on the rectified images.
                for (int i = 50; i < rect_combined.rows - 1; i += 50)
                {
                    cv::line(rect_combined, cv::Point(0, i), cv::Point(rect_combined.cols, i), cv::Scalar(0, 0, 255), 1);
                }

                cv::imshow("Horizontal Lines (Rectified)", rect_combined);
            }
        }

        /*while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }*/

        if (cv::waitKey(30) == 27) // Break on ESC
        {
            break;
        }
    }
}
