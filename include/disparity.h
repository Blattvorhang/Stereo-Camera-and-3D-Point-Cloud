#ifndef DISPARITY_H
#define DISPARITY_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>

class DisparityMapGenerator
{
public:
    enum DisparityMethod {
        StereoBM,
        SAD,
        SSD,
        NCC,
        DP,
        GC,
        BP,
        SGM,
        BM,
        SGBM,
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
    void computeSAD(int numDisparities = 16 * 5, int blockSize = 5);
    void computeSSD();
    void computeBM();
    void computeSGBM();
    void computeStereoBM();
    void preprocessImage(cv::Mat& image, bool useGaussianBlur = true);
    // 其他方法可以根据需要添加
};

#endif // DISPARITY_H
