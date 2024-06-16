#pragma once
#include <string>
#include <vector>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "utils.h"

#define TEST_EDGEDETECTORHYPERPARAMS 0

using namespace std;

cv::Mat sumOfKernelTo1(const cv::Mat kernel);
cv::Mat equalKernelWeigth(const cv::Mat &kernel);
cv::Mat getAvgKernel(int size);
cv::Mat zeroKernelCentre(cv::Mat kernel, float centreRadius);
cv::Mat getKernelWithRadiusConstraint(cv::Mat kernel, int radius, bool isSumTo1 = true);
cv::Mat discardTinyWeight(cv::Mat kernel, float ratio = 0.3);

// Compute the Gaussian gradient kernel
int computeGaussianSize(double sigma);
cv::Mat getGaussianGradientKernel(float sigma, float seta, float theta);
// Compute the ellipse Gaussian kernel
cv::Mat getEllipseGaussianKernel(float sigma_x, float sigma_y, float theta);
// Compute the Gabor kernel
cv::Mat getGaborKernel(cv::Size ksize, float sigma, float theta, float lambda, float gamma, float psi);
// Compute late_inhi_kernel
cv::Mat late_inhi_kernel(float sigma_x, float simga_y, int delta_degree, float theta);

// Compute kernels for edge thinning
void getTBsideKernels(int idx, cv::Mat &topK, cv::Mat &bottomK);

// Auxiliary function for visulizing the kernel with positive and negtive values
cv::Mat visuKernel(const cv::Mat &kernel);
// Auxiliary function for visulizing kernels for edge thinning
cv::Mat mergeTBsideKernels(const cv::Mat &topK, const cv::Mat &bottomK);

bool _isLeft(float Ax, float Ay, float Bx, float By, float Px, float Py);
bool _isRight(float Ax, float Ay, float Bx, float By, float Px, float Py);
cv::Mat constructPatternLineEp(int length, float orientation, int flag, int thickness=2);
cv::Mat getLineKernel(int length, float orientation, int flag, int thickness=2);

struct EdgeDetectorHyperparams
{
public:
    EdgeDetectorHyperparams();
    
    int rf_size_levels = 4; // scale number
    
    int channels = 6; // 1 channel or 3 channels
    bool ifSqrt = true; // For enhancing the contrast of image
    
    // These hyper-parameters are for the Gaussian gradient kernel, which is used for edge detection
    int numOrient = 12; // The number of orientation cells
    float sigmaGG = 1;
    float seta = 0.5;
    int GGKernelRadius = 2;
    vector<cv::Mat> GaussianGradientKernels;

    // These hyper-parameters are used to suppress texture edges
    int numLine = 12;
    vector<cv::Mat> same_faci_kernels;
    vector<cv::Mat> late_inhi_kernels;
    vector<cv::Mat> sm_kernels;
    cv::Mat full_inhi_kernel;
    // TGkernels
    int tg_scales = 5;
    vector<vector<cv::Mat>> TGkernels;

    // These hyper-parameters are for V2
    vector<cv::Mat> line_l5_kernels;
    vector<cv::Mat> line_l3_kernels;
    vector<cv::Mat> line_ep_left_kernels;
    vector<cv::Mat> line_ep_right_kernels;
    vector<cv::Mat> line_ep_kernels;
};
