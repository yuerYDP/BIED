#include "EdgeDetector.h"

cv::Mat compute_local_contrast(cv::Mat gray_img, float radius)
{
    cv::Mat avg_kernel = getAvgKernel(int(radius * 2));
    cv::Mat mean_img;
    cv::filter2D(gray_img, mean_img, gray_img.depth(), avg_kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::Mat variance_img;
    cv::pow(gray_img - mean_img, 2, variance_img);
    cv::Mat mean_variance_img;
    cv::filter2D(variance_img, mean_variance_img, variance_img.depth(), avg_kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
    cv::Mat std_deviation_img;
    cv::sqrt(mean_variance_img, std_deviation_img);
    cv::Mat local_contrast;
    cv::sqrt(std_deviation_img, local_contrast);
    return local_contrast;
}

cv::Mat EdgeDetector::getEdge()
{
    return this->edge;
}
cv::Mat EdgeDetector::getEdgeWoWeak()
{
    return this->edgeWoWeak;
}
cv::Mat EdgeDetector::getOrientEdgeEstimation()
{
    return this->orientEdgeEstimation;
}
cv::Mat EdgeDetector::getThinnedEdge()
{
    return this->thinnedEdge;
}
cv::Mat EdgeDetector::getEstiOptimalOrient()
{
    return this->estiOptimalOrient;
}
cv::Mat EdgeDetector::getColoredEstiOptimalOrient()
{
    return visulizeOptimalOrientation(this->estiOptimalOrient, this->edge);
}
cv::Mat EdgeDetector::getColoredEstiOptimalOrientGrid()
{
    return visulizeOptimalOrientationGrid(this->estiOptimalOrient, this->edge);
}

void EdgeDetector::RetinaLGN(cv::Mat &src)
{
    cv::Mat tSrc;
    src.convertTo(tSrc, CV_32F, 1/255.0); // Convert to CV_32F
    double maxV;
    cv::minMaxLoc(tSrc, 0, &maxV);
    tSrc = 1./ float(maxV) * tSrc;
    // Image enhancement
    if(this->hparams.ifSqrt)
    {
	    cv::Mat tempSqrt;
	    cv::sqrt(tSrc, tempSqrt);
	    tSrc = tempSqrt;
    }
    vector<cv::Mat> bgr;
    cv::split(tSrc, bgr);
    if(this->hparams.channels == 1)
    {
        parallelChannels.push_back(1 / 3. * (bgr[0] + bgr[1] + bgr[2]));
    }
    else if(this->hparams.channels == 6)
    {
    	this->parallelChannels.push_back(bgr[2] - bgr[1]);
    	this->parallelChannels.push_back(bgr[0] - 0.5 * (bgr[2] + bgr[1]));
    	this->parallelChannels.push_back(bgr[2] - 0.5 * bgr[1]);
    	this->parallelChannels.push_back(bgr[0] - 0.5 * 0.5 * (bgr[2] + bgr[1]));
    	cv::Mat lu;
    	lu = 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2];
        this->parallelChannels.push_back(lu);
        cv::Mat lc;
        lc = compute_local_contrast(lu, 2.5);
        this->parallelChannels.push_back(lc);
    }
}


vector<cv::Mat> EdgeDetector::V1_surround_modulation(vector<cv::Mat> orientation_edge)
{
    cv::Mat sum_edge = sumVectorMat(orientation_edge);
    cv::Mat avg_edge = 1. / this->hparams.numOrient * sum_edge;
    cv::Mat full_inhi;
    cv::filter2D(avg_edge, full_inhi, avg_edge.depth(), this->hparams.full_inhi_kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    
    vector<cv::Mat> orientation_edge_sm;
    for(int i = 0; i < this->hparams.numOrient; i++)
    {
        cv::Mat d_edge = orientation_edge[i];
        cv::Mat same_faci;
        cv::filter2D(d_edge, same_faci, d_edge.depth(), this->hparams.same_faci_kernels[i]);
    	cv::Mat late_inhi;
    	cv::filter2D(d_edge, late_inhi, d_edge.depth(), this->hparams.late_inhi_kernels[i]);
    	cv::Mat temp;
    	temp = d_edge + same_faci - 1 * late_inhi - 0.8 * full_inhi;
    	
    	cv::threshold(temp, temp, 0, 1.0, cv::THRESH_TOZERO);
    	orientation_edge_sm.push_back(temp);
    }
    
    return orientation_edge_sm;
}

vector<cv::Mat> EdgeDetector::V1_texture_boundary(cv::Mat texture_info)
{
    vector<cv::Mat> texture_boundary;
    for(int numO = 0; numO < 24; numO++)
    {
        cv::Mat t;
        for(int i = 0; i < this->hparams.tg_scales; i++)
        {
            cv::Mat temp;
            cv::filter2D(texture_info, temp, texture_info.depth(), this->hparams.TGkernels[i][numO]);
            
            float dis = float(- this->hparams.TGkernels[i][numO].cols / 2 / 2);
            cv::Mat M = cv::Mat::zeros(2, 3, CV_32F);
            M.at<float>(0, 0) = 1;
            M.at<float>(0, 2) = dis * cos(numO * 15. / 180. * CV_PI + CV_PI / 2);
            M.at<float>(1, 1) = 1;
            M.at<float>(1, 2) = dis * sin(numO * 15. / 180. * CV_PI + CV_PI / 2);
            cv::Mat temp_result;
            cv::warpAffine(temp, temp_result, M, cv::Size(temp.cols, temp.rows));
            
            if(i == 0)
            {
                t = temp_result;
            }
            else
            {
                t = t + temp_result;
            }
        }

        texture_boundary.push_back(t);
    }

    return texture_boundary;
}

cv::Mat EdgeDetector::V1_get_orient_edge(vector<cv::Mat> crf_response, cv::Size pri_shape)
{
    vector<cv::Mat> rf_response;
    rf_response = V1_surround_modulation(crf_response);
    cv::Mat rf_response_merged;
    cv::merge(rf_response, rf_response_merged);
    cv::Mat t_rf_response;
    cv::resize(rf_response_merged, t_rf_response, pri_shape, 0, 0);
    cv::Mat sc_edge;
    sc_edge = findMaxOfChannels(t_rf_response);
    double maxVal;
    cv::minMaxLoc(sc_edge, 0, &maxVal);
    sc_edge = float(1. / maxVal) * sc_edge;
    return sc_edge;
    
    cv::Mat texture_info = sumVectorMat(crf_response);
    vector<cv::Mat> texture_boundary_24;
    texture_boundary_24 = V1_texture_boundary(texture_info);
    cv::Mat texture_boundary_24_merged;
    cv::merge(texture_boundary_24, texture_boundary_24_merged);
    cv::Mat t_texture_boundary_24;
    cv::resize(texture_boundary_24_merged, t_texture_boundary_24, pri_shape, 0, 0);
    cv::Mat texture_boundary;
    texture_boundary = findMaxOfChannels(t_texture_boundary_24);
    
    double maxVal1;
    cv::minMaxLoc(sc_edge, 0, &maxVal1);
    float ratio1 = 7;
    cv::threshold(sc_edge, sc_edge, maxVal1/ratio1, maxVal1/ratio1, cv::THRESH_TRUNC);
    sc_edge = float(1. / (maxVal1 / ratio1)) * sc_edge;
    
    double maxVal2;
    cv::minMaxLoc(texture_boundary, 0, &maxVal2);
    float ratio2 = 7;
    cv::threshold(texture_boundary, texture_boundary, maxVal2/ratio2, maxVal2/ratio2, cv::THRESH_TRUNC);
    texture_boundary = float(1. / (maxVal2 / ratio2)) *  texture_boundary;
    
    cv::Mat t_edge;
    t_edge = sc_edge.mul(texture_boundary);
    double maxVal3;
    cv::minMaxLoc(t_edge, 0, &maxVal3);
    t_edge = float(1. / maxVal3) * t_edge;
    return t_edge;
}
vector<vector<cv::Mat>> EdgeDetector::V1()
{
    vector<vector<cv::Mat>> edge_response;
    for(int level = 0; level < this->hparams.rf_size_levels; level++)
    {
    	vector<cv::Mat> level_response;
    	for(int idx_c = 0; idx_c < this->hparams.channels; idx_c++)
    	{
            cv::Mat ch = this->parallelChannels[idx_c].clone();
            cv::Mat resizedCh;
            cv::resize(ch, resizedCh, cv::Size(), 1./pow(2, level), 1./pow(2, level));
            
            vector<cv::Mat> crf_response;
            for(int i = 0; i < this->hparams.numOrient; i++)
            {
                cv::Mat crf;
                cv::filter2D(resizedCh, crf, ch.depth(), hparams.GaussianGradientKernels[i], cv::Point(-1, -1), 0, cv::BORDER_REFLECT);  // BORDER_DEFAULT = BORDER_REFLECT_101
                crf = cv::abs(crf);
                crf_response.push_back(crf);
            }
            
            // Surround modulation and texture boundary fusion
            cv::Mat t_edge;
            t_edge = V1_get_orient_edge(crf_response, cv::Size(ch.cols, ch.rows));
            
            level_response.push_back(t_edge);
        }
        edge_response.push_back(level_response);
    }

    return edge_response;
}



cv::Mat EdgeDetector::V2_ep_modulation(cv::Mat ch_edge)
{
    vector<cv::Mat> line_response;
    for(int i = 0; i < this->hparams.numLine; i++)
    {
        cv::Mat t;
        cv::filter2D(ch_edge, t, ch_edge.depth(), this->hparams.line_l3_kernels[i], cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        line_response.push_back(t);
    }
    
    for(int i = 0; i < this->hparams.numLine; i++)
    {
        cv::Mat line_ep_left, line_ep_right, line_ep_inhi;
        //cv::filter2D(ch_edge, line_ep_left, ch_edge.depth(), this->hparams.line_ep_left_kernels[i], cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        //cv::filter2D(ch_edge, line_ep_right, ch_edge.depth(), this->hparams.line_ep_right_kernels[i], cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        //line_ep_inhi = cv::abs(line_ep_left - line_ep_right);
        cv::filter2D(ch_edge, line_ep_inhi, ch_edge.depth(), this->hparams.line_ep_kernels[i], cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        line_ep_inhi = cv::abs(line_ep_inhi);
        cv::Mat line_ep_inhi_response;
        cv::filter2D(line_ep_inhi, line_ep_inhi_response, line_ep_inhi.depth(), this->hparams.line_l3_kernels[i], cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        line_response[i] = line_response[i] - 2.0 * line_ep_inhi_response;
    }
    
    cv::Mat mergedLR;
    cv::merge(line_response, mergedLR);
    cv::Mat line_cc_edge;
    line_cc_edge = findMaxOfChannels(mergedLR);
    cv::threshold(line_cc_edge, line_cc_edge, 0, 1., cv::THRESH_TOZERO);
    return line_cc_edge;
}
vector<cv::Mat> EdgeDetector::V2(vector<vector<cv::Mat>> V1_response)
{
    vector<cv::Mat> result;
    for(int idx_c = 0; idx_c < this->hparams.channels; idx_c++)
    {
        cv::Mat t;
        for(int level = 0; level < this->hparams.rf_size_levels; level++)
         {
             cv::Mat ep_modulated = V2_ep_modulation(V1_response[level][idx_c]);
             if(level == 0)
             {
                 t = ep_modulated;
             }
             else
                 t = t + 1. / float(level + 1) * ep_modulated;
         }
         result.push_back(t);
    }
    return result;
}


void EdgeDetector::V4(vector<cv::Mat> V2_response)
{
    cv::Mat V4_response = sumVectorMat(V2_response);
    double maxV;
    cv::minMaxLoc(V4_response, 0, &maxV);
    this->edge = 1. / float(maxV) * V4_response;
}


void EdgeDetector::BIED(string fileName, cv::Mat &img)
{
    this->parallelChannels.clear();
    imageName = fileName;
    
    cout << "test" << endl;
    
    RetinaLGN(img);
    
    cout << "test" << endl;
    
    vector<vector<cv::Mat>> V1_response;
    V1_response = V1();
    
    cout << "test" << endl;
    
    vector<cv::Mat> V2_response;
    V2_response = V2(V1_response);
    
    V4(V2_response);
    
    cout << "complete" << endl;
    
    // Discard weak edges
    this->discardWeakEdge();
    //cout << "discardWeakEdge cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;

    // Edge thinning
    this->thinEdge();
    //cout << "thinEdge cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;
}


void EdgeDetector::discardWeakEdge()
{
    cv::threshold(this->edge, this->edgeWoWeak, 0, 1.0, cv::THRESH_TOZERO);
    //writeImage(this->edgeWoWeak, imageName + "_edgeWoWeak", false, "ed_visu");
}
void EdgeDetector::thinEdge()
{
    // 现在的问题是，我使用 V1 的 CC 来细化
    // 我的目标是，Gabor 函数对应的
    // edge 其实不大重要
    // 重要的是什么，是 thinnedEdge

    // Estimate the optimal orientation
    this->orientEdgeEstimation = filter2D_multiK(this->edgeWoWeak, hparams.orientKernels);
    this->edgeWoWeak = findMaxOfChannels(this->orientEdgeEstimation);
    this->estiOptimalOrient = findArgMaxOfChannels(this->orientEdgeEstimation);
    // Prepare the values of two sides
    cv::Mat topSideEstimation, bottomSideEstimation;
    topSideEstimation = filter2D_multiK(this->edgeWoWeak, this->hparams.topSideKernels);
    bottomSideEstimation = filter2D_multiK(this->edgeWoWeak, this->hparams.bottomSideKernels);
    // Edge thinning
    this->thinnedEdge = cv::Mat::zeros(this->edgeWoWeak.size(), CV_32F);
    int chns = this->hparams.numOrient_estiOri;
    int maxIdx = -1; // If there is no edge, set it to -1
    float curV = 0.0;
    float topV = 0.0;
    float bottomV = 0.0;
    for(int i = 0; i < this->edgeWoWeak.rows; i++)
    {
        for(int j = 0; j < this->edgeWoWeak.cols; j++)
        {
            maxIdx = this->estiOptimalOrient.at<int>(i, j);
            if(maxIdx != -1 && this->edgeWoWeak.at<float>(i, j) > 0)
            {
                curV = this->edgeWoWeak.at<float>(i, j);
                topV = topSideEstimation.ptr<float>(i)[j * chns + maxIdx];
                bottomV = bottomSideEstimation.ptr<float>(i)[j * chns + maxIdx];
                if(curV >= topV && curV >= bottomV)
                    this->thinnedEdge.at<float>(i, j) = curV;
            }
        }
    }
    
    cv::threshold(this->edgeWoWeak, this->edgeWoWeak, 0, 1.0, cv::THRESH_TOZERO);
    cv::threshold(this->thinnedEdge, this->thinnedEdge, 0.03, 1.0, cv::THRESH_TOZERO);
    
    //writeImage(this->thinnedEdge, imageName + "thinnedEdge", false, "ed_visu");
    cv::Mat thinnedEdgeBI;
    cv::threshold(thinnedEdge, thinnedEdgeBI, 0, 1.0, cv::THRESH_BINARY);
    writeImage(thinnedEdge, imageName, false, "thinnedEdge");
    //writeImage(this->getColoredEstiOptimalOrient(), imageName + "_edge_orient", false, "ed_visu");
    //writeImage(this->getColoredEstiOptimalOrientGrid(), imageName + "_edge_orientGrid", false, "ed_visu");
    // Binarize the thinned edge and visulize it
    cv::Mat t;
    cv::threshold(this->thinnedEdge, t, 0.01, 1.0, cv::THRESH_BINARY);
    writeImage(t, this->imageName, false, "thinnedEdgeBI");
    //writeImage(visulizeOptimalOrientation(this->estiOptimalOrient, t), this->imageName + "_thinnedEdge_optimalOrie", false, "ed_visu");
    //writeImage(visulizeOptimalOrientationGrid(this->estiOptimalOrient, t), this->imageName + "_thinnedEdge_optimalOrie_grid", false, "ed_visu");
}







