#include "../include/disparity.h"
#include <opencv2/calib3d.hpp>
#include "../include/semi_global_matching.h"
#include <chrono>
using namespace std::chrono;

DisparityMapGenerator::DisparityMapGenerator(const cv::Mat& leftImage, const cv::Mat& rightImage, DisparityMethod method)
    : left_image_(leftImage), right_image_(rightImage), method_(method) {
    preprocessImage(left_image_);
    preprocessImage(right_image_);
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
    cv::imshow("preprocess Left", left_image_);
    cv::imshow("preprocess Right", right_image_);
    switch (method_) {
    case BM:
        computeBM();
        break;
    case SGBM:
        computeSGBM();
        break;
    case NCC:
        computeNCC();
        break;
    case SGM:
        computeSGM();
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

void DisparityMapGenerator::computeNCC() {
    int windowSize = 21;  // 窗口大小
    int maxDisparity = 60;  // 最大视差
    int halfWindowSize = windowSize / 2;

    cv::Mat leftGray = left_image_.clone();
    cv::Mat rightGray = right_image_.clone();

    disparity_ = cv::Mat::zeros(leftGray.size(), CV_32F);

    // 遍历图像
    for (int y = halfWindowSize; y < leftGray.rows - halfWindowSize; ++y) {
        for (int x = halfWindowSize; x < leftGray.cols - halfWindowSize; ++x) {
            double maxNCC = -1.0;
            int bestShift = 0;

            // 左图的窗口
            cv::Rect leftRect(x - halfWindowSize, y - halfWindowSize, windowSize, windowSize);
            cv::Mat leftROI = leftGray(leftRect);
            double leftMean = cv::mean(leftROI)[0];

            for (int shift = 0; shift < maxDisparity; ++shift) {
                int rightX = x + shift;
                if (rightX + halfWindowSize >= rightGray.cols) break;

                // 右图的窗口
                cv::Rect rightRect(rightX - halfWindowSize, y - halfWindowSize, windowSize, windowSize);
                cv::Mat rightROI = rightGray(rightRect);
                double rightMean = cv::mean(rightROI)[0];

                // 计算归一化互相关
                double num = 0, den1 = 0, den2 = 0;
                for (int dy = -halfWindowSize; dy <= halfWindowSize; ++dy) {
                    for (int dx = -halfWindowSize; dx <= halfWindowSize; ++dx) {
                        double lPixel = leftGray.at<uchar>(y + dy, x + dx) - leftMean;
                        double rPixel = rightGray.at<uchar>(y + dy, rightX + dx) - rightMean;
                        num += lPixel * rPixel;
                        den1 += lPixel * lPixel;
                        den2 += rPixel * rPixel;
                    }
                }
                double den = sqrt(den1 * den2);
                double ncc = (den == 0) ? 0 : num / den;

                if (ncc > maxNCC) {
                    maxNCC = ncc;
                    bestShift = shift;
                }
            }
            disparity_.at<float>(y, x) = static_cast<float>(bestShift);
        }
    }
}

void DisparityMapGenerator::computeSGM() {
    const int width = left_image_.cols;
    const int height = right_image_.rows;

    // 左右影像的灰度数据
    auto bytes_left = new uint8_t[width * height];
    auto bytes_right = new uint8_t[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes_left[i * width + j] = left_image_.at<uint8_t>(i, j);
            bytes_right[i * width + j] = right_image_.at<uint8_t>(i, j);
        }
    }

    printf("Loading Views...Done!\n");

    // SGM匹配参数设计
    SemiGlobalMatching::SGMOption sgm_option;
    // 聚合路径数
    sgm_option.num_paths = 8;
    // 候选视差范围
    sgm_option.min_disparity = 0;
    sgm_option.max_disparity = 64;
    // census窗口类型
    sgm_option.census_size = SemiGlobalMatching::Census5x5;
    // 一致性检查
    sgm_option.is_check_lr = true;
    sgm_option.lrcheck_thres = 1.0f;
    // 唯一性约束
    sgm_option.is_check_unique = true;
    sgm_option.uniqueness_ratio = 0.99;
    // 剔除小连通区
    sgm_option.is_remove_speckles = true;
    sgm_option.min_speckle_aera = 50;
    // 惩罚项P1、P2
    sgm_option.p1 = 10;
    sgm_option.p2_init = 150;
    // 视差图填充
    // 视差图填充的结果并不可靠，若工程，不建议填充，若科研，则可填充
    sgm_option.is_fill_holes = false;

    printf("w = %d, h = %d, d = [%d,%d]\n\n", width, height, sgm_option.min_disparity, sgm_option.max_disparity);

    // 定义SGM匹配类实例
    SemiGlobalMatching sgm;

    //···············································································//
    // 初始化
    printf("SGM Initializing...\n");
    auto start = std::chrono::steady_clock::now();
    if (!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM初始化失败！" << std::endl;
        return;
    }
    auto end = std::chrono::steady_clock::now();
    auto tt = duration_cast<std::chrono::milliseconds>(end - start);
    printf("SGM Initializing Done! Timing : %lf s\n\n", tt.count() / 1000.0);

    //···············································································//
    // 匹配
    printf("SGM Matching...\n");
    start = std::chrono::steady_clock::now();
    // disparity数组保存子像素的视差结果
    auto disparity = new float[uint32_t(width * height)]();
    if (!sgm.Match(bytes_left, bytes_right, disparity)) {
        std::cout << "SGM匹配失败！" << std::endl;
        return;
    }
    end = std::chrono::steady_clock::now();
    tt = duration_cast<std::chrono::milliseconds>(end - start);
    printf("\nSGM Matching...Done! Timing :   %lf s\n", tt.count() / 1000.0);

    //···············································································//
    // 显示视差图
    // 注意，计算点云不能用disp_mat的数据，它是用来显示和保存结果用的。计算点云要用上面的disparity数组里的数据，是子像素浮点数
    cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
    float min_disp = width, max_disp = -width;
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            const float disp = disparity[i * width + j];
            if (disp != Invalid_Float) {
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            const float disp = disparity[i * width + j];
            if (disp == Invalid_Float) {
                disp_mat.data[i * width + j] = 0;
            }
            else {
                disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }

    disparity_ = disp_mat;
    delete[] disparity;
    disparity = nullptr;
    delete[] bytes_left;
    bytes_left = nullptr;
    delete[] bytes_right;
    bytes_right = nullptr;

}