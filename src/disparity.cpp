// TODO: Disparity optimization
// TODO: Disparity refinement

#include "../include/disparity.h"
#include <opencv2/calib3d.hpp>

DisparityMapGenerator::DisparityMapGenerator(const cv::Mat& leftImage, const cv::Mat& rightImage, DisparityMethod method) {
	// 初始化视差计算器
    this->left_image_ = leftImage;
    this->right_image_ = rightImage;
    this->method_ = method;
}

// 函数对单个图像进行预处理
void DisparityMapGenerator::preprocessImage(cv::Mat& image, bool useGaussianBlur) {
    // 将图像转换为灰度图（如果尚未转换）
    if (image.channels() > 1) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }

    // 调整图像大小到640x360
    cv::resize(image, image, cv::Size(640, 360));

    // 应用高斯模糊以减少图像噪声（如果启用）
    if (useGaussianBlur) {
        // 使用更大的高斯核增强模糊效果
        cv::GaussianBlur(image, image, cv::Size(3, 3), 0);
    }

    // 直方图均衡化以增强图像对比度
    cv::equalizeHist(image, image);

    // 转换图像格式到8位无符号整数类型（如果尚未转换）
    if (image.type() != CV_8U) {
        image.convertTo(image, CV_8U, 255.0);
    }
}

void DisparityMapGenerator::computeDisparity() {
    preprocessImage(left_image_, true);
    preprocessImage(right_image_, true);
    cv::imshow("preprocess Left", left_image_);
    cv::imshow("preprocess Right", right_image_);
    switch (method_) {
    case SAD:
        computeSAD();
        break;
    case SSD:
        computeSSD();
        break;
    case BM:
        computeBM();
        break;
    case SGBM:
        computeSGBM();
        break;
        // 其他方法可以根据需要添加
    default:
        throw std::invalid_argument("Unsupported disparity method");
    }
}

void DisparityMapGenerator::displayDisparity() {
    // 将视差图从CV_16S转换到CV_8U
    cv::Mat disp8;
    double minVal, maxVal;
    cv::minMaxLoc(disparity_, &minVal, &maxVal);
    disparity_.convertTo(disp8, CV_8U, 255 / (maxVal - minVal), -minVal * 255 / (maxVal - minVal));

    // 应用伪彩色映射增强视觉效果
    cv::Mat dispColor;
    cv::applyColorMap(disp8, dispColor, cv::COLORMAP_JET);

    // 应用双边滤波增强视差图
    cv::Mat dispBilateral;
    cv::bilateralFilter(dispColor, dispBilateral, 9, 75, 75);

    // 显示处理后的视差图
    cv::namedWindow("Enhanced Disparity Map", cv::WINDOW_NORMAL);
    cv::imshow("Enhanced Disparity Map", dispBilateral);
}


void DisparityMapGenerator::computeSAD(int numDisparities, int blockSize) {
    // 计算绝对差异和（SAD）视差图
    disparity_ = cv::Mat(left_image_.size(), CV_8U, cv::Scalar(0));

    // 对于每个像素，在允许的视差范围内搜索最佳匹配
    for (int y = blockSize / 2; y < left_image_.rows - blockSize / 2; ++y) {
        for (int x = blockSize / 2; x < left_image_.cols - blockSize / 2; ++x) {
            int minSAD = INT_MAX;
            int bestDisparity = 0;
            for (int d = 0; d < numDisparities; ++d) {
                int SAD = 0;
                if (x - d < blockSize / 2) continue;  // 避免左图超出边界

                for (int dy = -blockSize / 2; dy <= blockSize / 2; ++dy) {
                    for (int dx = -blockSize / 2; dx <= blockSize / 2; ++dx) {
                        int leftPixel = left_image_.at<uchar>(y + dy, x + dx);
                        int rightX = x + dx - d;
                        if (rightX < 0 || rightX >= right_image_.cols) continue; // 避免右图超出边界
                        int rightPixel = right_image_.at<uchar>(y + dy, rightX);
                        SAD += std::abs(leftPixel - rightPixel);
                    }
                }

                if (SAD < minSAD) {
                    minSAD = SAD;
                    bestDisparity = d;
                }
            }
            disparity_.at<uchar>(y, x) = static_cast<uchar>((bestDisparity * 255) / (numDisparities - 1)); // 改进视差归一化
        }
    }
}



void DisparityMapGenerator::computeSSD() {
    // 计算平方差异（SD）视差图
    cv::Mat disparity;
    // SD方法的实现
    // TODO: 实现SD方法
    this->disparity_ = disparity;
}

void DisparityMapGenerator::computeBM() {
    // 使用块匹配（Block Matching）方法计算视差图
    cv::Mat disparity;
    int numDisparities = 16 * 5;  // 视差范围
    int blockSize = 15;  // 块大小
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(numDisparities, blockSize);
    bm->compute(left_image_, right_image_, disparity);
    this->disparity_ = disparity;
}

void DisparityMapGenerator::computeSGBM() {
    // 使用半全局块匹配（Semi-Global Block Matching）方法计算视差图
    cv::Mat disparity;
    int numDisparities = 16 * 5;  // 视差范围
    int blockSize = 16;  // 块大小
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, numDisparities, blockSize);
    sgbm->compute(left_image_, right_image_, disparity);
    this->disparity_ = disparity;
}
