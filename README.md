# Stereo Camera and 3D Point Cloud

## Task Description
Establish a stereo camera system for generating a real-time depth-map and 3D point cloud.

## Guide
### 1. Establish a stereo camera system
### 2. Calibrate the cameras as a stereo system
Use OpenCV stereo camera calibration tools.

Refer to: https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_calib.cpp

### 3. Create stereo images and generate feature pairs
There are lots of works to do:

- Compensate the distortion from lens
- Utilizing with epipolar geometry
- Generating projection matrix with `cv::stereoRectify()`

Find the same feature point in left/right image.

### 4. Generating disparity map
Calculate disparity from the matches.

Addition: You can try `cv::StereoBM()` and `cv::StereoSGBM()` for comparison.

Refer to: https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_match.cpp

### 5. Calculate depth map
Calculate depth from disparity with `cv::reprojectImageTo3D()`

## Extension Task
1. Create a 3D point cloud
2. Improve accuracy/quality of depth map
3. Try other algorithms to improve speed/accuracy/quality of depth map

## Process
1. Calibrate the stereo camera (intrinsic matrices)
2. Rectify the stereo camera
3. Keypoint detection and matching
4. Obtain the fundamental matrix $\mathbf{F}$ using the 8-point algorithm
5. Compute the essential matrix $\mathbf{E}$
6. Decompose the essential matrix $\mathbf{E}$ to get the rotation matrix $\mathbf{R}$ and the translation vector $\mathbf{t}$
7. Stereopsis
8. Disparity map
9. Triangulation to get the 3D point cloud

## UML
![](./stereo_system.png)