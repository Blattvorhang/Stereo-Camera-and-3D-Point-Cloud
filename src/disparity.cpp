#include "../include/disparity.h"
#include "../include/semi_global_matching.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <chrono>
#include <thread>
#include <vector>
#include <limits>
#include <omp.h>
using namespace std::chrono;

DisparityMapGenerator::DisparityMapGenerator(const cv::Mat& leftImage, const cv::Mat& rightImage, DisparityMethod method)
    : left_image_(leftImage), right_image_(rightImage), method_(method) {
    numDisparities_ = ((left_image_.cols / 8) + 15) & -16;  // 视差搜索范围
}

// 函数对单个图像进行预处理
void DisparityMapGenerator::preprocessImage(cv::Mat& image, bool useGaussianBlur) {

    // 调整图像大小
    //cv::resize(image, image, cv::Size((int)image.cols / 2, (int)image.rows / 2));

    // 将图像转换为灰度图
    if (image.channels() > 1) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }

    // 转换图像格式到8位无符号整数类型
    if (image.type() != CV_8U) {
        image.convertTo(image, CV_8U);
    }

    // 自适应直方图均衡化以增强图像对比度
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4.0);
    clahe->apply(image, image);

    // 应用高斯模糊以减少图像噪声
    if (useGaussianBlur) {
        cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
    }

    // 边缘保留滤波
    cv::Mat temp;
    cv::ximgproc::guidedFilter(image, image, temp, 8, 0.1);
    image = temp;

    // 亮度均匀化 (使用形态学闭操作进行背景提取和亮度均匀化)
    cv::Mat background;
    cv::morphologyEx(image, background, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    cv::Mat uniformLight;
    cv::absdiff(image, background, uniformLight);
    image = uniformLight;
}

void DisparityMapGenerator::computeDisparity(cv::Mat &disparity) {
    preprocessImage(left_image_);
	preprocessImage(right_image_);
    cv::imshow("preprocess Left", left_image_);
    cv::imshow("preprocess Right", right_image_);
    switch (method_) {
    case BM:
        computeBM();
        break;
    case SGBM:
        computeSGBM();
        break;
    case SGM:
        computeSGM();
        break;
    default:
        throw std::invalid_argument("Unsupported disparity method");
    }
    enhanceSubpixel();
    // 将视差图转换为 CV_32F
    disparity.convertTo(disparity, CV_32F, 1.0 / 16.0); // 假设视差图最初是 CV_16S 类型

    // 重建右图像
    cv::Mat reconstructedRightImage = reconstructRightImage(left_image_, disparity);

    // 计算光度一致性误差
    double mse = computePhotometricConsistencyMSE(reconstructedRightImage, right_image_);
    double mae = computePhotometricConsistencyMAE(reconstructedRightImage, right_image_);

    std::cout << "Photometric Consistency MSE: " << mse << std::endl;
    std::cout << "Photometric Consistency MAE: " << mae << std::endl;
    disparity = disparity_;
}

void DisparityMapGenerator::displayDisparity() {
    // 检查视差图是否为空
    if (disparity_.empty()) {
        std::cerr << "Disparity map is empty!" << std::endl;
        return;
    }

    // 将视差图从CV_32F转换到CV_8U
    cv::Mat disp8;
    double minVal, maxVal;
    cv::minMaxLoc(disparity_, &minVal, &maxVal);
    disparity_.convertTo(disp8, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // 计算积分图和对应的像素点个数图像
    cv::Mat integralImage, pixelCount;
    cv::integral(disp8, integralImage, pixelCount, CV_32S, CV_32S);

    // 多层次均值滤波填充空洞
    cv::Mat dispFilled = disp8.clone();
    int windowSize = 128; // 初始较大窗口尺寸
    while (windowSize >= 3) {
        cv::Mat temp = dispFilled.clone();
        for (int i = 0; i < dispFilled.rows; i++) {
            for (int j = 0; j < dispFilled.cols; j++) {
                if (dispFilled.at<uchar>(i, j) == 0) {
                    int x1 = std::max(i - windowSize / 2, 0);
                    int y1 = std::max(j - windowSize / 2, 0);
                    int x2 = std::min(i + windowSize / 2, dispFilled.rows - 1);
                    int y2 = std::min(j + windowSize / 2, dispFilled.cols - 1);

                    int count = pixelCount.at<int>(x2 + 1, y2 + 1) - pixelCount.at<int>(x1, y2 + 1) - pixelCount.at<int>(x2 + 1, y1) + pixelCount.at<int>(x1, y1);
                    if (count > 0) {
                        int sum = integralImage.at<int>(x2 + 1, y2 + 1) - integralImage.at<int>(x1, y2 + 1) - integralImage.at<int>(x2 + 1, y1) + integralImage.at<int>(x1, y1);
                        temp.at<uchar>(i, j) = sum / count;
                    }
                }
            }
        }
        dispFilled = temp;
        windowSize /= 2; // 缩小窗口尺寸
    }

    // 应用闭运算（膨胀后腐蚀）填充空洞
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat dispClosed;
    cv::morphologyEx(dispFilled, dispClosed, cv::MORPH_CLOSE, kernel);

    // 应用引导滤波增强边缘
    cv::Mat dispGuided;
    cv::ximgproc::guidedFilter(dispClosed, dispClosed, dispGuided, 9, 75);

    // 应用伪彩色映射增强视觉效果
    cv::Mat dispColor;
    cv::applyColorMap(dispGuided, dispColor, cv::COLORMAP_JET);

    // 显示处理后的视差图
    cv::imshow("Enhanced Disparity Map", dispColor);
}

void DisparityMapGenerator::displayLRCheckResult() {
    // 检查 lrCheckedDisparity_ 是否为空
    if (lrCheckedDisparity_.empty()) {
        std::cerr << "LR Checked Disparity is empty!" << std::endl;
        return;
    }

    // 创建一个与 lrCheckedDisparity_ 相同大小的单通道图像
    cv::Mat result = cv::Mat::zeros(lrCheckedDisparity_.size(), CV_8UC1);

    for (int y = 0; y < lrCheckedDisparity_.rows; ++y) {
        for (int x = 0; x < lrCheckedDisparity_.cols; ++x) {
            if (lrCheckedDisparity_.at<float>(y, x) != -1.0f) {
                result.at<uchar>(y, x) = 255; // 有效视差值显示为白色
            }
            else {
                result.at<uchar>(y, x) = 0; // 无效视差值显示为黑色
            }
        }
    }

    // 显示结果
    cv::imshow("LR Check Disparity", result);
}

void DisparityMapGenerator::computeBM() {
    if (left_image_.empty() || right_image_.empty()) {
        std::cerr << "One or both input images are empty!" << std::endl;
        return;
    }

    int numDisparities = 16 * 5;
    int blockSize = 15;

    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities, blockSize);
    stereoBM->compute(left_image_, right_image_, disparity_);

}

void DisparityMapGenerator::computeSGBM() {
    cv::Mat leftDisparity, rightDisparity;
    int winSize = 5;  // 块大小
    int numDisparities = ((left_image_.cols / 8) + 15) & -16;  // 视差搜索范围

    cv::Ptr<cv::StereoSGBM> sgbmLeft = cv::StereoSGBM::create(0, numDisparities, winSize);
    sgbmLeft->setPreFilterCap(31);
    sgbmLeft->setBlockSize(winSize);
    sgbmLeft->setP1(8 * winSize * winSize);
    sgbmLeft->setP2(32 * winSize * winSize);
    sgbmLeft->setMinDisparity(0);
    sgbmLeft->setNumDisparities(numDisparities);
    sgbmLeft->setUniquenessRatio(10);
    sgbmLeft->setSpeckleWindowSize(100);
    sgbmLeft->setSpeckleRange(32);
    sgbmLeft->setDisp12MaxDiff(1);
    sgbmLeft->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    sgbmLeft->compute(left_image_, right_image_, leftDisparity);

    cv::Ptr<cv::StereoSGBM> sgbmRight = cv::StereoSGBM::create(0, numDisparities, winSize);
    sgbmRight->setPreFilterCap(31);
    sgbmRight->setBlockSize(winSize);
    sgbmRight->setP1(8 * winSize * winSize);
    sgbmRight->setP2(32 * winSize * winSize);
    sgbmRight->setMinDisparity(0);
    sgbmRight->setNumDisparities(numDisparities);
    sgbmRight->setUniquenessRatio(10);
    sgbmRight->setSpeckleWindowSize(100);
    sgbmRight->setSpeckleRange(32);
    sgbmRight->setDisp12MaxDiff(1);
    sgbmRight->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    sgbmRight->compute(right_image_, left_image_, rightDisparity);

    this->disparity_ = leftDisparity;
    this->right_disparity_ = rightDisparity;
}

void DisparityMapGenerator::computeSGM() {
    // 获取左右图像的宽度和高度
    const int width = left_image_.cols;
    const int height = right_image_.rows;

    // 分配内存存储左右图像的灰度数据
    auto bytes_left = new uint8_t[width * height];
    auto bytes_right = new uint8_t[width * height];

    // 将图像数据从cv::Mat转换为灰度数组
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes_left[i * width + j] = left_image_.at<uint8_t>(i, j);
            bytes_right[i * width + j] = right_image_.at<uint8_t>(i, j);
        }
    }

    // 输出加载图像完成的信息
    printf("Loading Views...Done!\n");

    // 配置SGM算法的参数
    SemiGlobalMatching::SGMOption sgm_option;
    sgm_option.num_paths = 8;  // 聚合路径数，决定了多路径聚合的数量
    sgm_option.min_disparity = 0;  // 最小视差值
    sgm_option.max_disparity = 64;  // 最大视差值
    sgm_option.census_size = SemiGlobalMatching::Census5x5;  // Census转换的窗口大小
    sgm_option.is_check_lr = true;  // 是否进行左右一致性检查
    sgm_option.lrcheck_thres = 1.0f;  // 左右一致性检查的阈值
    sgm_option.is_check_unique = true;  // 是否进行唯一性检查
    sgm_option.uniqueness_ratio = (float)0.99;  // 唯一性检查的比例阈值
    sgm_option.is_remove_speckles = true;  // 是否移除小区块
    sgm_option.min_speckle_aera = 50;  // 最小的连通区块大小
    sgm_option.p1 = 10;  // SGM算法的惩罚项P1
    sgm_option.p2_init = 150;  // SGM算法的惩罚项P2
    sgm_option.is_fill_holes = false;  // 是否填充视差图中的孔洞（通常用于科研）

    // 输出配置的视差范围信息
	std::cout << "w = " << width << ", h = " << height << ", d = [" << sgm_option.min_disparity << "," << sgm_option.max_disparity << "]" << std::endl;

    // 创建SGM匹配类的实例
    SemiGlobalMatching sgm;

    // 初始化SGM算法
    std::cout << "SGM Initializing..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    if (!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM初始化失败！" << std::endl;
        return;
    }
    auto end = std::chrono::steady_clock::now();
    auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "SGM Initializing Done! Timing : " << tt.count() / 1000.0 << " s" << std::endl;

    // 执行SGM匹配
	std::cout << "SGM Matching..." << std::endl;
    start = std::chrono::steady_clock::now();
    auto disparity = new float[uint32_t(width * height)]();  // 存储子像素精度的视差结果
    if (!sgm.Match(bytes_left, bytes_right, disparity)) {
        std::cout << "SGM匹配失败！" << std::endl;
        return;
    }
    end = std::chrono::steady_clock::now();
    tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "SGM Matching...Done! Timing : " << tt.count() / 1000.0 << " s" << std::endl;

    // 生成视差图，用于显示和结果保存
    cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
    float min_disp = (float)width, max_disp = (float)-width;
    // 计算视差图的最小和最大视差值
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            const float disp = disparity[i * width + j];
            if (disp != Invalid_Float) {  // Invalid_Float为无效值的标识
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    // 根据计算得到的最小和最大视差值归一化视差数据到0-255范围内
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            const float disp = disparity[i * width + j];
            if (disp == Invalid_Float) {
                disp_mat.data[i * width + j] = 0;  // 无效视差值设置为0
            }
            else {
                disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }

    // 保存最终的视差图
    disparity_ = disp_mat;

    // 释放内存资源
    delete[] disparity;
    delete[] bytes_left;
    delete[] bytes_right;
}

void DisparityMapGenerator::applyLRCheck() {
    const int width = disparity_.cols;
    const int height = disparity_.rows;

    // 将 disparity_ 和 right_disparity_ 转换为 CV_32F 类型
    cv::Mat leftDisparity, rightDisparity;
    disparity_.convertTo(leftDisparity, CV_32F);
    right_disparity_.convertTo(rightDisparity, CV_32F);

    lrCheckedDisparity_ = leftDisparity.clone();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float d = leftDisparity.at<float>(y, x);
            if (x - d < 0 || x - d >= width) {
                lrCheckedDisparity_.at<float>(y, x) = -1; // 标记为无效
            }
            else {
                float d_right = rightDisparity.at<float>(y, x - static_cast<int>(d));
                if (std::abs(d - d_right) > 1) {
                    lrCheckedDisparity_.at<float>(y, x) = -1; // 标记为无效
                }
            }
        }
    }
}

float DisparityMapGenerator::computeCost(int x, int y, float d) {
    int xr = x - static_cast<int>(std::round(d));  // 使用四舍五入将浮点数转换为整数
    if (xr < 0 || xr >= right_image_.cols) {
        return std::numeric_limits<float>::max(); // 超出图像边界的高代价
    }
    return std::abs(static_cast<float>(left_image_.at<uchar>(y, x)) - static_cast<float>(right_image_.at<uchar>(y, xr)));
}

void DisparityMapGenerator::enhanceSubpixel() {
    // 确保视差图被转换为 CV_32F 类型
    if (disparity_.type() != CV_32F) {
        disparity_.convertTo(disparity_, CV_32F);
    }

    const int width = disparity_.cols;
    const int height = disparity_.rows;

    // 创建一个新的矩阵用于存储增强后的视差
    cv::Mat enhancedDisparity = disparity_.clone();

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float d = disparity_.at<float>(y, x);
            if (d < 0) continue; // 跳过无效视差

            // 计算左右相邻视差的代价
            float c1 = computeCost(x, y, d - 1.0f);
            float c2 = computeCost(x, y, d);
            float c3 = computeCost(x, y, d + 1.0f);

            // 计算分母
            float denominator = c1 - 2 * c2 + c3;
            if (std::abs(denominator) > std::numeric_limits<float>::epsilon()) {
                // 计算亚像素校正
                float delta = (c1 - c3) / (2 * denominator);
                enhancedDisparity.at<float>(y, x) = d + delta;
            }
        }
    }

    // 更新视差图
    disparity_ = enhancedDisparity;
}

// 重建右图像
cv::Mat DisparityMapGenerator::reconstructRightImage(const cv::Mat& leftImage, const cv::Mat& disparity) {
    // 检查视差图是否为空
    if (disparity_.empty()) {
        std::cerr << "Disparity map is empty!" << std::endl;
        return cv::Mat();
    }

    // 确保视差图为 CV_32F 类型
    cv::Mat disparity32F;
    if (disparity_.type() != CV_32F) {
        std::cerr << "Converting disparity to CV_32F" << std::endl;
        disparity_.convertTo(disparity32F, CV_32F);
    }
    else {
        disparity32F = disparity_;
    }

    // 调试信息
    std::cout << "Disparity type: " << disparity32F.type() << " (Expected: " << CV_32F << ")" << std::endl;
    std::cout << "Disparity size: " << disparity32F.size() << std::endl;

    cv::Mat rightImage = cv::Mat::zeros(leftImage.size(), leftImage.type());

    for (int y = 0; y < leftImage.rows; ++y) {
        for (int x = 0; x < leftImage.cols; ++x) {
            float d = disparity32F.at<float>(y, x);
            int xr = static_cast<int>(x - d + 0.5f); // 使用四舍五入到最近的整数
            if (xr >= 0 && xr < leftImage.cols) {
                rightImage.at<uchar>(y, xr) = leftImage.at<uchar>(y, x);
            }
        }
    }

    return rightImage;
}

// 计算光度一致性的均方误差
double DisparityMapGenerator::computePhotometricConsistencyMSE(const cv::Mat& reconstructedRightImage, const cv::Mat& actualRightImage) {
    cv::Mat diff;
    cv::absdiff(reconstructedRightImage, actualRightImage, diff);
    diff = diff.mul(diff);
    return cv::mean(diff)[0];
}

// 计算光度一致性的平均绝对误差
double DisparityMapGenerator::computePhotometricConsistencyMAE(const cv::Mat& reconstructedRightImage, const cv::Mat& actualRightImage) {
    cv::Mat diff;
    cv::absdiff(reconstructedRightImage, actualRightImage, diff);
    return cv::mean(diff)[0];
}