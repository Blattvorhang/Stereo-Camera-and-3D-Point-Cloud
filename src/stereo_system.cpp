#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "../include/stereo_system.h"

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


void StereoSystem::checkSize(const cv::Mat& mat, int expected_rows, int expected_cols) {
    if (mat.rows != expected_rows || mat.cols != expected_cols) {
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
    try {
        checkSize(cameraMatrixL, 3, 3); // Assuming camera matrices are 3x3
        checkSize(distCoeffsL, 1, 5);   // Assuming distortion coefficients are 1x5
        checkSize(cameraMatrixR, 3, 3); 
        checkSize(distCoeffsR, 1, 5); 
        checkSize(R_, 3, 3);             // Rotation matrix is 3x3
        checkSize(T_, 3, 1);             // Translation vector is 3x1
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
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

void StereoSystem::run()
{
    // Open video stream from camera.
    cv::VideoCapture cap(camera_id_);
    if (!cap.isOpened())
    {
        throw std::runtime_error("Could not open camera.");
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width_ * 2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height_);

    calibrateStereoCameras();

    cv::Mat ori_left, ori_right;    // original images
    cv::Mat rect_left, rect_right;  // stereo rectified images

    while (true)
    {
        captureImages(cap, ori_left, ori_right);
        rectifyImages(ori_left, ori_right, rect_left, rect_right);

        // Show rectified images.
        cv::imshow("Rectified Left", rect_left);
        cv::imshow("Rectified Right", rect_right);

        // TODO: Compute disparity map.


        // Show additional debug/educational figures.
        if (enable_debug_)
        {
            // if (!frame.empty())      { cv::imshow("Original", frame); }
            if (!ori_left.empty())  { cv::imshow("Original Left", ori_left); }
            if (!ori_right.empty()) { cv::imshow("Original Right", ori_right); }
        }

        // Update the windows.
        cv::waitKey(1);
    }

    /*
    // Process each image in the dataset.
    for (DataElement element{}; dataset >> element;)
    {
      // Todo 1: Convert geographical position and attitude to local Cartesian pose.
      // Compute the pose of the body in the local coordinate system.
      // DONE 1.1: Finish Attitude::toQuaternion().
      const Sophus::SE3d pose_local_body = local_system.toLocalPose(element.body_position_in_geo,
                                                                    element.body_attitude_in_geo.toSO3());

      // Add body coordinate axis to the 3D viewer.
      // DONE 1.2: Write line of code below to add body to viewer.
      viewer.addBody(pose_local_body, element.img_num);

      // Todo 2: Compute the pose of the camera
      // Compute the pose of the camera relative to the body.
      // DONE 2.1: Finish CartesianPosition::toVector().
      // DONE 2.2: Construct pose_body_camera correctly using element.
      const Sophus::SE3d pose_body_camera{element.camera_attitude_in_body.toSO3(),
                                          element.camera_position_in_body.toVector()};

      // Compute the pose of the camera relative to the local coordinate system.
      // DONE 2.3: Construct pose_local_camera correctly using the poses above.
      const Sophus::SE3d pose_local_camera{pose_local_body * pose_body_camera};

      // Todo 3: Undistort the images.
      // Construct a camera model based on the intrinsic calibration and camera pose.
      // DONE 3.1: Finish Intrinsics::toCalibrationMatrix().
      // DONE 3.2: Finish Intrinsics::toDistortionVector().
      const PerspectiveCameraModel camera_model{element.intrinsics.toCalibrationMatrix(),
                                               pose_local_camera,
                                               element.intrinsics.toDistortionCoefficientVector()};

      // Undistort image.
      // DONE 3.3: Undistort image using the camera model. Why should we undistort the image?
      // Undistort the original image, instead of using it directly.
      cv::Mat undistorted_img = camera_model.undistortImage(element.image);

      // Todo 4: Project a geographic world point into the images
      // Project world point (the origin) into the image.
      // DONE 4.1: Finish PerspectiveCameraModel::computeCameraProjectionMatrix().
      // DONE 4.2: Finish PerspectiveCameraModel::projectWorldPoint().
      // DONE 4.3: Optionally finish PerspectiveCameraModel::projectWorldPoints().
      const Eigen::Vector2d pix_pos = camera_model.projectWorldPoint(Eigen::Vector3d::Zero());

      // Draw a marker in the image at the projected position.
      const Eigen::Vector2i pix_pos_int = (pix_pos.array().round()).cast<int>();
      cv::drawMarker(undistorted_img, {pix_pos_int.x(), pix_pos_int.y()}, {0.,0.,255.}, cv::MARKER_CROSS, 40, 3);

      // Show the image.
      // DONE: Write line of code below to show the image with the marker.
      cv::imshow(window_name_, undistorted_img);
      // Add the camera to the 3D viewer.
      // DONE 4.4: Write line of code below to add body to viewer.
      viewer.addCamera(camera_model, undistorted_img, element.img_num);
      // Update the windows.
      viewer.spinOnce();
      cv::waitKey(100);
    }

    // Remove image viewer.
    cv::destroyWindow(window_name_);
    */
}
