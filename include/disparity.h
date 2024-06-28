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
        SAD,
        SSD,
        NCC,
        DP,
        GC,
        BP,
        SGM,
        FBS
    };
    DisparityMapGenerator(const cv::Mat& leftImage, const cv::Mat& rightImage, DisparityMethod method);
    void computeDisparity();
    void displayDisparity();  // ���Ӳ�ͼ��һ������ʾ
private:
    cv::Mat left_image_;
    cv::Mat right_image_;
    cv::Mat disparity_;
    DisparityMethod method_;
    void computeSAD(int numDisparities = 16 * 5, int blockSize = 5);
    void computeSSD();
    void computeBM();
    void computeSGBM();
    void preprocessImage(cv::Mat& image, bool useGaussianBlur = true);
    // �����������Ը�����Ҫ����
};

#endif // DISPARITY_H