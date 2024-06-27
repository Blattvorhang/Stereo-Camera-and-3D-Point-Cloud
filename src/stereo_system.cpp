#include "../include/stereo_system.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

StereoSystem::StereoSystem(const std::string &data_path)
    : data_path_{data_path}, window_name_{"World point in camera"}
{}

void StereoSystem::run()
{
    // Construct viewers.
    cv::namedWindow(window_name_);

    // Open video stream from camera.
    const int camera_id = 0;
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened())
    {
        throw std::runtime_error("Could not open camera.");
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280 * 2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Assuming cap is your cv::VideoCapture object
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Current Resolution: " << width << "x" << height << std::endl;

    cv::Mat frame;
    cv::Mat leftImage, rightImage;

    while (true)
    {
        // Capture frame from camera.
        cap >> frame;

        // Cut the frame into two images.
        leftImage = frame.colRange(0, frame.cols / 2);
        rightImage = frame.colRange(frame.cols / 2, frame.cols);

        // Show left and right images.
        cv::imshow("Left", leftImage);
        cv::imshow("Right", rightImage);

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
