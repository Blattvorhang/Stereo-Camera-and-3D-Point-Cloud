#include "../include/disparity.h"
#include "../include/semi_global_matching.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
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

    // 调整图像大小到640x360（或你认为合适的大小）
    cv::resize(image, image, cv::Size(), 0.5, 0.5);

    // 应用高斯模糊以减少图像噪声（如果启用）
    if (useGaussianBlur) {
        cv::GaussianBlur(image, image, cv::Size(5, 5), 0);
    }

    // 自适应直方图均衡化以增强图像对比度
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4.0);
    clahe->apply(image, image);

    // 转换图像格式到8位无符号整数类型（如果尚未转换）
    if (image.type() != CV_8U) {
        image.convertTo(image, CV_8U);
    }

    // 边缘保留滤波
    cv::Mat temp;
    cv::ximgproc::guidedFilter(image, image, temp, 8, 0.1);
    image = temp;
}


void DisparityMapGenerator::computeDisparity(cv::Mat &disparity) {
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

    disparity = disparity_;
}

void DisparityMapGenerator::displayDisparity() {
    // 将视差图从CV_16S转换到CV_8U
    cv::Mat disp8;
    double minVal, maxVal;
    cv::minMaxLoc(disparity_, &minVal, &maxVal);
    disparity_.convertTo(disp8, CV_8U, 255 / (maxVal - minVal), -minVal * 255 / (maxVal - minVal));

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
    // cv::namedWindow("Enhanced Disparity Map", cv::WINDOW_NORMAL);
    cv::imshow("Enhanced Disparity Map", dispColor);
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
    int winSize = 10;
    int numDisparities = ((left_image_.cols / 8) + 15) & -16;  // 视差搜索范围
    int blockSize = winSize;  // 块大小

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, numDisparities, blockSize);
    sgbm->setPreFilterCap(31);  // 预处理滤波器截断值
    sgbm->setBlockSize(blockSize);  // SAD窗口大小
    sgbm->setP1(8 * winSize * winSize);  // 控制视差平滑度第一参数
    sgbm->setP2(32 * winSize * winSize);  // 控制视差平滑度第二参数
    sgbm->setMinDisparity(0);  // 最小视差
    sgbm->setNumDisparities(numDisparities);  // 视差搜索范围
    sgbm->setUniquenessRatio(10);  // 视差唯一性百分比
    sgbm->setSpeckleWindowSize(200);  // 检查视差连通区域变化度的窗口大小
    sgbm->setSpeckleRange(1);  // 视差变化阈值
    sgbm->setDisp12MaxDiff(0);  // 左右视差图最大容许差异
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);  // 采用全尺寸双通道动态编程算法

    sgbm->compute(left_image_, right_image_, disparity);
    this->disparity_ = disparity;
}


void DisparityMapGenerator::computeNCC() {

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
