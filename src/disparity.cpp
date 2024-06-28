// TODO: Disparity optimization
// TODO: Disparity refinement

#include "../include/disparity.h"
#include <opencv2/calib3d.hpp>

DisparityMapGenerator::DisparityMapGenerator(const cv::Mat& leftImage, const cv::Mat& rightImage, DisparityMethod method) {
	// ��ʼ���Ӳ������
    this->left_image_ = leftImage;
    this->right_image_ = rightImage;
    this->method_ = method;
}

// �����Ե���ͼ�����Ԥ����
void DisparityMapGenerator::preprocessImage(cv::Mat& image, bool useGaussianBlur) {
    // ��ͼ��ת��Ϊ�Ҷ�ͼ�������δת����
    if (image.channels() > 1) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }

    // ����ͼ���С��640x360
    cv::resize(image, image, cv::Size(640, 360));

    // Ӧ�ø�˹ģ���Լ���ͼ��������������ã�
    if (useGaussianBlur) {
        // ʹ�ø���ĸ�˹����ǿģ��Ч��
        cv::GaussianBlur(image, image, cv::Size(3, 3), 0);
    }

    // ֱ��ͼ���⻯����ǿͼ��Աȶ�
    cv::equalizeHist(image, image);

    // ת��ͼ���ʽ��8λ�޷����������ͣ������δת����
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
        // �����������Ը�����Ҫ���
    default:
        throw std::invalid_argument("Unsupported disparity method");
    }
}

void DisparityMapGenerator::displayDisparity() {
    // ���Ӳ�ͼ��CV_16Sת����CV_8U
    cv::Mat disp8;
    double minVal, maxVal;
    cv::minMaxLoc(disparity_, &minVal, &maxVal);
    disparity_.convertTo(disp8, CV_8U, 255 / (maxVal - minVal), -minVal * 255 / (maxVal - minVal));

    // Ӧ��α��ɫӳ����ǿ�Ӿ�Ч��
    cv::Mat dispColor;
    cv::applyColorMap(disp8, dispColor, cv::COLORMAP_JET);

    // Ӧ��˫���˲���ǿ�Ӳ�ͼ
    cv::Mat dispBilateral;
    cv::bilateralFilter(dispColor, dispBilateral, 9, 75, 75);

    // ��ʾ�������Ӳ�ͼ
    cv::namedWindow("Enhanced Disparity Map", cv::WINDOW_NORMAL);
    cv::imshow("Enhanced Disparity Map", dispBilateral);
}


void DisparityMapGenerator::computeSAD(int numDisparities, int blockSize) {
    // ������Բ���ͣ�SAD���Ӳ�ͼ
    disparity_ = cv::Mat(left_image_.size(), CV_8U, cv::Scalar(0));

    // ����ÿ�����أ���������ӲΧ���������ƥ��
    for (int y = blockSize / 2; y < left_image_.rows - blockSize / 2; ++y) {
        for (int x = blockSize / 2; x < left_image_.cols - blockSize / 2; ++x) {
            int minSAD = INT_MAX;
            int bestDisparity = 0;
            for (int d = 0; d < numDisparities; ++d) {
                int SAD = 0;
                if (x - d < blockSize / 2) continue;  // ������ͼ�����߽�

                for (int dy = -blockSize / 2; dy <= blockSize / 2; ++dy) {
                    for (int dx = -blockSize / 2; dx <= blockSize / 2; ++dx) {
                        int leftPixel = left_image_.at<uchar>(y + dy, x + dx);
                        int rightX = x + dx - d;
                        if (rightX < 0 || rightX >= right_image_.cols) continue; // ������ͼ�����߽�
                        int rightPixel = right_image_.at<uchar>(y + dy, rightX);
                        SAD += std::abs(leftPixel - rightPixel);
                    }
                }

                if (SAD < minSAD) {
                    minSAD = SAD;
                    bestDisparity = d;
                }
            }
            disparity_.at<uchar>(y, x) = static_cast<uchar>((bestDisparity * 255) / (numDisparities - 1)); // �Ľ��Ӳ��һ��
        }
    }
}



void DisparityMapGenerator::computeSSD() {
    // ����ƽ�����죨SD���Ӳ�ͼ
    cv::Mat disparity;
    // SD������ʵ��
    // TODO: ʵ��SD����
    this->disparity_ = disparity;
}

void DisparityMapGenerator::computeBM() {
    // ʹ�ÿ�ƥ�䣨Block Matching�����������Ӳ�ͼ
    cv::Mat disparity;
    int numDisparities = 16 * 5;  // �ӲΧ
    int blockSize = 15;  // ���С
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(numDisparities, blockSize);
    bm->compute(left_image_, right_image_, disparity);
    this->disparity_ = disparity;
}

void DisparityMapGenerator::computeSGBM() {
    // ʹ�ð�ȫ�ֿ�ƥ�䣨Semi-Global Block Matching�����������Ӳ�ͼ
    cv::Mat disparity;
    int numDisparities = 16 * 5;  // �ӲΧ
    int blockSize = 16;  // ���С
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, numDisparities, blockSize);
    sgbm->compute(left_image_, right_image_, disparity);
    this->disparity_ = disparity;
}
