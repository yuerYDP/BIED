#pragma once
#include <string>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "utils.h"
#include "EdgeDetectorHyperparams.h"

#define TEST_EDGEDETECTOR 1

using namespace std;

cv::Mat compute_local_contrast(cv::Mat gray_img, float radius);

class EdgeDetector
{
public:
    EdgeDetector(){}

    // getter
    cv::Mat getEdge();

//private:
    EdgeDetectorHyperparams hparams;
    string imageName;
    cv::Mat edge;

    vector<cv::Mat> parallelChannels;
    
    void RetinaLGN(cv::Mat &src);
    
    vector<cv::Mat> V1_surround_modulation(vector<cv::Mat> orientation_edge);
    vector<cv::Mat> V1_texture_boundary(cv::Mat texture_info);
    cv::Mat V1_get_orient_edge(vector<cv::Mat> crf_response, cv::Size pri_shape);
    vector<vector<cv::Mat>> V1();
    
    cv::Mat V2_ep_modulation(cv::Mat ch_edge);
    vector<cv::Mat> V2(vector<vector<cv::Mat>> V1_response);
    
    void V4(vector<cv::Mat> V2_response);
    
    void BIED(string fileName, cv::Mat &src);
    
};

