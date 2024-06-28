#ifndef DISPARITY_H
#define DISPARITY_H

#include <opencv2/opencv.hpp>

class DisparityMapGenerator
{
public:
    //cv::Mat computeDisparity(cv::Mat leftImage, cv::Mat rightImage, DisparityMethod method);

    enum DisparityMethod {
        NCC,
        BM,
        SGBM
    };
};

#endif // DISPARITY_H
