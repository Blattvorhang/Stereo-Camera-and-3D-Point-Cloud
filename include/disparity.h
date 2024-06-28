#ifndef DISPARITY_H
#define DISPARITY_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>

class DisparityMapGenerator
{
public:
    //cv::Mat computeDisparity(cv::Mat leftImage, cv::Mat rightImage, DisparityMethod method);

    enum DisparityMethod {
        BM,
        SGBM,
        NCC,
        DP,
        GC,
        BP,
        SGM,
        FBS
    };
    DisparityMapGenerator(const cv::Mat& leftImage, const cv::Mat& rightImage, DisparityMethod method);
    void computeDisparity();
    void displayDisparity();  // 将视差图归一化并显示
private:
    cv::Mat left_image_;
    cv::Mat right_image_;
    cv::Mat disparity_;
    DisparityMethod method_;
    void computeBM();
    void computeSGBM();
    void computeNCC();
    void computeSGM();
    void preprocessImage(cv::Mat& image, bool useGaussianBlur = true);
    // 其他方法可以根据需要添加
};

#endif // DISPARITY_H
