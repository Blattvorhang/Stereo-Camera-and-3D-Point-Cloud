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
1. Calibrate the stereo camera (using MATLAB)
   1. Intrinsic matrices $\mathbf{K}_L$ and $\mathbf{K}_R$
   2. Obtain the *fundamental matrix* $\mathbf{F}$ using the 8-point algorithm
   3. Compute the *essential matrix* $\mathbf{E}$
   4. Decompose the essential matrix $\mathbf{E}$ to get the *rotation matrix* $\mathbf{R}$ and the *translation vector* $\mathbf{t}$ ($\mathbf{p}_R^C = \mathbf{R}\mathbf{p}_L^C+\mathbf{t}$)
2. Rectify the stereo camera (lens undistortion and stereo rectification)
3. Stereo matching (correspondence pair search on the same image row)
   1. Block matching (BM)
   2. Semi-global block matching (SGBM)
4.  Disparity map (left and right image), optimization and refinement
5.  Triangulation to get the 3D point cloud

## Debug Images
1. Original images
2. Rectified images
3. Disparity map
4. Depth map
5. 3D point cloud
6. 3D point cloud with color

## UML
```mermaid
classDiagram
    class Camera {
        +Camera();
        +cv::Mat getIntrinsicMatrix() const;
        +cv::Mat getDistortionCoeffs() const;
        +cv::Mat getProjectionMatrix() const;
        +void setRectificationMatrices(const cv::Mat &R, const cv::Mat &P);
        -cv::Mat K_;
        -cv::Mat distortion_coeffs_;
        -cv::Mat projection_matrix_;
        -cv::Mat rectified_R_;
        -cv::Mat rectified_P_;
    }

    class StereoSystem {
        -Camera leftCamera
        -Camera rightCamera
        -cv::Mat stereoRectificationMap1
        -cv::Mat stereoRectificationMap2
        -Eigen::Matrix4f projectionMatrix1
        -Eigen::Matrix4f projectionMatrix2
        +StereoCameraSystem(Camera left, Camera right)
        +void calibrate()
        +std::pair<cv::Mat, cv::Mat> rectifyImages(cv::Mat leftImage, cv::Mat rightImage)
        +cv::Mat generateDisparityMap(cv::Mat leftImage, cv::Mat rightImage, DisparityMethod method)
        +cv::Mat generateDepthMap(cv::Mat disparityMap)
    }

    class DisparityMapGenerator {
        +cv::Mat computeDisparity(cv::Mat leftImage, cv::Mat rightImage, DisparityMethod method)
        +enum DisparityMethod;
        +DisparityMapGenerator(const cv::Mat& leftImage, const cv::Mat& rightImage, DisparityMethod method);
        +void computeDisparity(cv::Mat &disparity);
        +void displayDisparity();
        +void displayLRCheckResult();
        -cv::Mat left_image_;
        -cv::Mat right_image_;
        -cv::Mat disparity_;
        -cv::Mat right_disparity_;
        -cv::Mat lrCheckedDisparity_;
        -int numDisparities_;
        -DisparityMethod method_;
        -void computeBM();
        -void computeSGBM();
        -void computeSGM();
        -void preprocessImage(cv::Mat& image, bool useGaussianBlur = true);
        -void applyLRCheck();
        -void enhanceSubpixel();
        -float computeCost(int x, int y, float d);
        -cv::Mat reconstructRightImage(const cv::Mat& leftImage, const cv::Mat& disparity);
        -double computePhotometricConsistencyMSE(const cv::Mat& reconstructedRightImage, const cv::Mat& actualRightImage);
        -double computePhotometricConsistencyMAE(const cv::Mat& reconstructedRightImage, const cv::Mat& actualRightImage);
    }

    class SemiGlobalMatching {
        +enum CensusSize;
        +struct SGMOption;
        +bool Initialize(const int32_t& width, const int32_t& height, const SGMOption& option);
        +bool Match(const uint8_t* img_left, const uint8_t* img_right, float* disp_left);
        +bool Reset(const uint32_t& width, const uint32_t& height, const SGMOption& option);
        -cv::Mat left_image_;
        -cv::Mat right_image_;
        -cv::Mat disparity_;
        -cv::Mat right_disparity_;
        -cv::Mat lrCheckedDisparity_;
        -int numDisparities_;
        -DisparityMethod method_;
        -void computeBM();
        -void computeSGBM();
        -void computeSGM();
        -void preprocessImage(cv::Mat& image, bool useGaussianBlur = true);
        -void applyLRCheck();
        -void enhanceSubpixel();
        -float computeCost(int x, int y, float d);
        -cv::Mat reconstructRightImage(const cv::Mat& leftImage, const cv::Mat& disparity);
        -double computePhotometricConsistencyMSE(const cv::Mat& reconstructedRightImage, const cv::Mat& actualRightImage);
        -double computePhotometricConsistencyMAE(const cv::Mat& reconstructedRightImage, const cv::Mat& actualRightImage);
    }

    class sgm_util {
        +void census_transform_5x5(const uint8_t* source, uint32_t* census, const int32_t& width, const int32_t& height);
        +void census_transform_9x7(const uint8_t* source, uint64_t* census, const int32_t& width, const int32_t& height);
        +uint8_t Hamming32(const uint32_t& x, const uint32_t& y);
        +uint8_t Hamming64(const uint64_t& x, const uint64_t& y);
        +void CostAggregateLeftRight(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,const int32_t& p1,const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
        +void CostAggregateUpDown(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
        +void CostAggregateDagonal_1(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
        +void CostAggregateDagonal_2(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);
        +void MedianFilter(const float* in, float* out, const int32_t& width, const int32_t& height, const int32_t wnd_size);
        +void RemoveSpeckles(float* disparity_map, const int32_t& width, const int32_t& height, const int32_t& diff_insame,const uint32_t& min_speckle_aera, const float& invalid_val);
    }

    StereoSystem --> Camera
    StereoSystem --> DisparityMapGenerator
    DisparityMapGenerator --> SemiGlobalMatching
    DisparityMapGenerator --> SemiGlobalMatching
    SemiGlobalMatching --> sgm_util
```

## References
- [HBVCAM-W202011HD V33](https://detail.1688.com/offer/753903056520.html)
- [Open Stereo Camera with OpenCV](https://zhaoxuhui.top/blog/2018/12/03/OpenSteroCameraWithOpenCV.html)
- [Stereo Camera Calibration (from MATLAB to OpenCV)](https://zhuanlan.zhihu.com/p/153329285)
- [Stereo Camera Calibration](https://www.cnblogs.com/champrin/p/17034043.html)
- [Depth Map in Stereo Matching](https://www.cnblogs.com/riddick/p/8486223.html)
- [cv::reprojectTo3D](https://blog.csdn.net/Gordon_Wei/article/details/86319058)
- [PCL Segmentation Fault](https://blog.csdn.net/weixin_45802055/article/details/131194547)
- [Middlebury Dataset](https://vision.middlebury.edu/stereo/data/)
- [Challenges of Stereo Matching](https://blog.csdn.net/He3he3he/article/details/101148558)