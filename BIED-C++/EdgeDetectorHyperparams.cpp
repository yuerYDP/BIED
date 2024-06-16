#include "EdgeDetectorHyperparams.h"

cv::Mat sumOfKernelTo1(const cv::Mat kernel)
{
    return kernel / cv::sum(kernel)[0];
}
cv::Mat equalKernelWeigth(const cv::Mat &kernel)
{
    // Set the positive weights and the absolute values of negative weights to equal to 1
    float sumP = 0.;
    float sumN = 0.;
    cv::Mat temp = kernel.clone();
    for(int i = 0; i < temp.rows; i++)
    {
        for(int j = 0; j < temp.cols; j++)
        {
            if(temp.at<float>(i, j) > 0)
                sumP += temp.at<float>(i, j);
            if(temp.at<float>(i, j) < 0)
                sumN += temp.at<float>(i, j);
        }
    }
    for(int i = 0; i < temp.rows; i++)
    {
        for(int j = 0; j < temp.cols; j++)
        {
            if(temp.at<float>(i, j) > 0)
                temp.at<float>(i, j) /= sumP;
            if(temp.at<float>(i, j) < 0)
                temp.at<float>(i, j) /= -sumN;
        }
    }
    return temp;
}
cv::Mat getAvgKernel(int size)
{
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F);
    float centre = float(size / 2);
    for(int i = 0; i < size; i++)
    {
        for(int j = 0; j < size; j++)
        {
            float dis = sqrt((i - centre) * (i - centre) + (j - centre) * (j - centre));
            if(dis > centre)
                kernel.at<float>(i, j) = 0;
        }
    }
    return kernel / cv::sum(kernel)[0];
}
cv::Mat zeroKernelCentre(cv::Mat kernel, float centreRadius)
{
    // Only consider kernel.depth() == CV_32F
    int centre = kernel.rows / 2;
    cv::Mat result = cv::Mat::zeros(kernel.size(), CV_32F);
    for(int i = 0; i < kernel.rows; i++)
    {
        for(int j = 0; j < kernel.cols; j++)
        {
            if((i - centre) * (i - centre) + (j - centre) * (j - centre) <= centreRadius * centreRadius)
            {
                result.at<float>(i, j) = 0;
            }
            else
                result.at<float>(i, j) = kernel.at<float>(i, j);
        }
    }
    return result;
}
cv::Mat getKernelWithRadiusConstraint(cv::Mat kernel, int radius, bool isSumTo1)
{
    cv::Mat result;
    int halfKernelSize= kernel.rows / 2;
    if(halfKernelSize <= radius)
    {
        result = kernel;
    }
    else
    {
        for(int i = 0; i < kernel.rows; i++)
        {
            for(int j = 0; j < kernel.cols; j++)
            {
                int dx = i - halfKernelSize;
                int dy = j - halfKernelSize;
                if(dx * dx + dy * dy > radius * radius)
                    kernel.at<float>(i, j) = 0;
            }
        }
        cv::Rect roiRect(halfKernelSize - radius, halfKernelSize - radius, 2 * radius + 1, 2 * radius + 1);
        result = kernel(roiRect).clone();
    }
    if(isSumTo1) // Set the sum of kernel to 1 according to the need
        result = sumOfKernelTo1(result);
    return result;
}
cv::Mat discardTinyWeight(cv::Mat kernel, float ratio)
{
    // Discard the tiny values of kernel
    double maxV;
    cv::minMaxLoc(kernel, 0, &maxV);
    maxV += 1e-4;
    for(int i = 0; i < kernel.rows; i++)
    {
        for(int j = 0; j < kernel.cols; j++)
        {
            if(abs(kernel.at<float>(i, j) / maxV) < ratio)
                kernel.at<float>(i, j) = 0;
        }
    }
    return kernel;
}
int computeGaussianSize(double sigma)
{
    double th = 1e-3;
    double k = 2 * sigma * sigma;
    int radius = int(sqrt(-k * log(th * sqrt(k * CV_PI))));
    return 2 * radius + 1;
}
cv::Mat getGaussianGradientKernel(float sigma, float seta, float theta)
{
    int ks = computeGaussianSize(sigma);
    cv::Mat kernel(ks, ks, CV_32F);

    int center = ks / 2;
    float disX, disY, x, y;
    for(int i = 0; i < ks; i++)
    {
        for(int j = 0; j < ks; j++)
        {
            disX = j - center;
            disY = i - center;
            x = disX * cos(theta) + disY * sin(theta);
            y = - disX * sin(theta) + disY * cos(theta);
            kernel.at<float>(i, j) = - x * exp(- (x * x + seta * seta * y * y) / (2 * sigma * sigma)) / (CV_PI * sigma * sigma);
        }
    }
    return kernel;
}
cv::Mat getEllipseGaussianKernel(float sigma_x, float sigma_y, float theta)
{
    int ks = computeGaussianSize(max(sigma_x, sigma_y));
    cv::Mat kernel(ks, ks, CV_32F);

    float centre_x = ks / 2;
    float centre_y = ks / 2;
    float a = pow(cos(theta), 2) / (2 * sigma_x * sigma_x) + pow(sin(theta), 2) / (2 * sigma_y * sigma_y);
    float b = -sin(2 * theta) / (4 * sigma_x * sigma_x) + sin(2 * theta) / (4 * sigma_y * sigma_y);
    float c = pow(sin(theta), 2) / (2 * sigma_x * sigma_x) + pow(cos(theta), 2) / (2 * sigma_y * sigma_y);

    for(int i = 0; i < ks; i ++)
    {
        for(int j = 0; j < ks; j++)
        {
            float x = i - centre_x;
            float y = j - centre_y;
            float t = exp(- (a * pow(x, 2) + 2 * b * x * y + c * pow(y, 2)));
            kernel.at<float>(i, j) = t;
        }
    }
    kernel = kernel / cv::sum(kernel)[0];
    return kernel;
}
cv::Mat getGaborKernel(int ksize, float sigma, float theta, float lambda, float gamma, float psi)
{
    cv::Mat kernel(ksize, ksize, CV_32F);
    float sigmaX = sigma;
    float sigmaY = sigma / gamma;
    float c = cos(theta);
    float s = sin(theta);
    // 确定卷积核函数的变化范围
    int xmax = ksize / 2;
    int ymax = ksize / 2;
    int xmin = -xmax;
    int ymin = -ymax;

    float ex = -0.5 / (sigmaX * sigmaX);
    float ey = -0.5 / (sigmaY * sigmaY);
    float cscale =  CV_PI * 2 / lambda;

    float _x = 0;
    float _y = 0;
    for(int y = ymin; y <= ymax; y++)
    {
        for(int x = xmin; x <= xmax; x++)
        {
            _x = x * c + y * s;
            _y = -x * s + y * c;
            float val = exp(ex * _x * _x + ey * _y * _y) * cos(cscale * _x + psi);
            kernel.at<float>(ymax + y, xmax + x) = val;
        }
    }
    return kernel;
}
cv::Mat late_inhi_kernel(float sigma_x, float sigma_y, int delta_degree, float theta)
{
    cv::Mat orient_kernel = getEllipseGaussianKernel(sigma_x, sigma_y, theta);
    int n_degree = 180 / delta_degree;
    vector<cv::Mat> orient_kernels;
    for(int i = 0; i < 3 * n_degree; i++)
    {
        float  t_theta = float(i * delta_degree) / 3.0 / 180.0;
        cv::Mat k = getEllipseGaussianKernel(sigma_x, sigma_y, t_theta * CV_PI);
        k = k - orient_kernel;
        cv::threshold(k, k, 0, 1.0, cv::THRESH_TOZERO);
        orient_kernels.push_back(k);
    }
    cv::Mat t_merged;
    cv::merge(orient_kernels, t_merged);
    cv::Mat t_max_response;
    t_max_response = findMaxOfChannels(t_merged);
    t_max_response = t_max_response / cv::sum(t_max_response)[0];
    return t_max_response;
}

void getTBsideKernels(int i, cv::Mat &topK, cv::Mat &bottomK)
{
    topK = cv::Mat::zeros(3, 3, CV_32F);
    bottomK = cv::Mat::zeros(3, 3, CV_32F);
    float theta = float(i) / 12 * CV_PI;
    if(0 <= i && i < 3) // [0, 45)
    {
        topK.at<float>(2, 2) = tan(theta);
        bottomK.at<float>(0, 0) = tan(theta);
        topK.at<float>(1, 2) = 1 - tan(theta);
        bottomK.at<float>(1, 0) = 1 - tan(theta);
    }
    else if(3 <= i && i < 6) // [45, 90)
    {
        theta = CV_PI / 2 - theta;
        topK.at<float>(2, 2) = tan(theta);
        bottomK.at<float>(0, 0) = tan(theta);
        topK.at<float>(2, 1) = 1 - tan(theta);
        bottomK.at<float>(0, 1) = 1 - tan(theta);
    }
    else if(6 <= i && i < 9) // [90, 135)
    {
        theta = theta - CV_PI / 2;
        topK.at<float>(2, 0) = tan(theta);
        bottomK.at<float>(0, 2) = tan(theta);
        topK.at<float>(2, 1) = 1 - tan(theta);
        bottomK.at<float>(0, 1) = 1 - tan(theta);
    }
    else // [135, 180)
    {
        theta = CV_PI - theta;
        topK.at<float>(2, 0) = tan(theta);
        bottomK.at<float>(0, 2) = tan(theta);
        topK.at<float>(1, 0) = 1 - tan(theta);
        bottomK.at<float>(1, 2) = 1 - tan(theta);
    }
}

cv::Mat visuKernel(const cv::Mat &kernel)
{
    cv::Mat result = cv::Mat::zeros(kernel.size(), CV_32FC3);
    float t;
    for(int i = 0; i < kernel.rows; i++)
    {
        for(int j = 0; j < kernel.cols; j++)
        {
            t = kernel.at<float>(i, j);
            if(t > 0)
                result.at<cv::Vec3f>(i, j)[2] = t;
            else if(t < 0)
                result.at<cv::Vec3f>(i, j)[1] = -t;
        }
    }
    return result;
}
cv::Mat mergeTBsideKernels(const cv::Mat &topK, const cv::Mat &bottomK)
{
    cv::Mat result = cv::Mat::zeros(topK.size(), CV_32FC3);
    for(int i = 0; i < topK.rows; i++)
    {
        for(int j = 0; j < topK.cols; j++)
        {
            result.at<cv::Vec3f>(i, j)[2] = topK.at<float>(i,j);
            result.at<cv::Vec3f>(i, j)[1] = bottomK.at<float>(i,j);
        }
    }
    cv::Mat resizedResult;
    cv::resize(result, resizedResult, cv::Size(), 16, 16);
    return resizedResult;
}

bool _isLeft(float Ax, float Ay, float Bx, float By, float Px, float Py)
{
    return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax) > 1e-4;
}
bool _isRight(float Ax, float Ay, float Bx, float By, float Px, float Py)
{
    return (Bx - Ax) * (Py - Ay) - (By - Ay) * (Px - Ax) < -1e-4;
}
cv::Mat constructPatternLineEp(int length, float orientation, int flag, int thickness)
{
    int centre = length / 2;
    float theta = orientation / 180 * CV_PI;
    float A, B, C;
    A = sin(theta);
    B = - cos(theta);
    C = - (A * centre + B * centre);

    float shiftDis = 10.0; // The shift for determining the left or right sides
    float shiftX = centre + cos(theta + CV_PI / 2) * shiftDis;
    float shiftY = centre + sin(theta + CV_PI / 2) * shiftDis;

    float cpX, cpY;
    float disToLine, disToLineCentre;
    cv::Mat kernel = cv::Mat::zeros(length, length, CV_32F);
    for(int i = 0; i < length; i++)
    {
        for(int j = 0; j < length; j++)
        {
            // Compute the distance to the line
            // d = (A * x0 + B * y0 + C) / sqrt(A**2 + B**2)
            disToLine = abs(A * i + B * j + C);
            // Get the prejection on the line
            // x = (B**2 * x0 - A * B * y0 - A * C) / (A**2 + B**2)
            // y = (- A * B * x0 + A**2 * y0 - B * C) / (A**2 + B**2)
            cpX = (B * B * i - A * B * j - A * C);
            cpY = (- A * B * i + A * A * j - B * C);
            // Compute the distance between "the prejection point" and "the middle point of the line"
            disToLineCentre = sqrt(pow(cpX - centre, 2) + pow(cpY - centre, 2));

            if(disToLineCentre <= length && disToLine <= thickness)
            {
            	if(flag == 0)
            	    kernel.at<float>(i, j) = 1;
            	else if(flag == 1 and _isLeft(float(centre), float(centre), float(shiftX), float(shiftY), float(i), float(j)) == true)
            	    kernel.at<float>(i, j) = 1;
            	else if(flag == 2 and _isRight(float(centre), float(centre), float(shiftX), float(shiftY), float(i), float(j)) == true)
            	    kernel.at<float>(i, j) = 1;
            }
        }
    }
    if(flag == 0)
    {
        kernel = (1. / cv::sum(kernel)[0]) * kernel;
    }
    else kernel = (0.5 / cv::sum(kernel)[0]) * kernel;
    return kernel;
}

cv::Mat getLineKernel(int length, float orientation, int flag, int thickness)
{
    cv::Mat kernel = cv::Mat::zeros(length, length, CV_32F);
    int grid_size = 101;
    
    cv::Mat kernel_grid = constructPatternLineEp(length * grid_size, orientation, flag);
    
    for(int i = 0; i < length; i++)
    {
        for(int j = 0; j < length; j++)
        {
            cv::Mat region = kernel_grid(cv::Rect(j * grid_size, i * grid_size, grid_size, grid_size)).clone();
            kernel.at<float>(i, j) = cv::sum(region)[0];
            kernel.at<float>(i, j) = kernel.at<float>(i, j) / grid_size / grid_size / thickness;
        }
    }
    return kernel;
}


EdgeDetectorHyperparams::EdgeDetectorHyperparams()
{
    float theta;
    // Get guassianGradientKernels
    for(int i = 0; i < this->numOrient; i++)
    {
        theta = float(i) / this->numOrient * CV_PI;
        cv::Mat kernel = getGaussianGradientKernel(this->sigmaGG, this->seta, theta);
        this->GaussianGradientKernels.push_back(kernel);
    }

    // Get same_faci_kernels late_inhi_kernels
    for(int i = 0; i < this->numOrient; i++)
    {
        float theta = float(i) / this->numOrient * CV_PI;
        cv::Mat sf_kernel;
        sf_kernel = getEllipseGaussianKernel(0.3, 2.0, theta);
        int centre = sf_kernel.rows / 2;
        sf_kernel.at<float>(centre, centre) = 0;
        this->same_faci_kernels.push_back(sf_kernel);
        
        cv::Mat li_kernel;
        li_kernel = late_inhi_kernel(0.3, 2.0, 180 / this->numOrient, theta);
        li_kernel.at<float>(centre, centre) = 0;
        this->late_inhi_kernels.push_back(li_kernel);
        sm_kernels.push_back(sf_kernel - li_kernel);
    }
    // Get full_inhi_kernel
    this->full_inhi_kernel = getAvgKernel(same_faci_kernels[0].rows);
    
    // Get TGkernels
    float radiuses[5] = {3.5, 5.5, 8.5, 12.5, 17.5};
    //float radiuses[1] = {3.5};
    for(int r = 0; r < 5; r++)
    {
        int size = int(radiuses[r] * 2);
        cv::Mat kernel = getAvgKernel(size);
        int centre = size / 2;
        
        cv::Mat top_half_kernel = cv::Mat::zeros(size, size, CV_32F);
        cv::Mat bottom_half_kernel = cv::Mat::zeros(size, size, CV_32F);
        for(int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++)
            {
                if(i < centre)
                    top_half_kernel.at<float>(i, j) = kernel.at<float>(i, j);
                if(i > centre)
                    bottom_half_kernel.at<float>(i, j) = kernel.at<float>(i, j);
            }
        }
        
        vector<cv::Mat> oKs;
        for(int idx = 0; idx < 24; idx++)
        {
            cv::Mat topM;
            topM = cv::getRotationMatrix2D(cv::Point2f(centre, centre), - idx * 15, 1.0);
            cv::Mat tK;
            cv::warpAffine(top_half_kernel, tK, topM, cv::Size(size, size));
            
            cv::Mat botM;
            botM = cv::getRotationMatrix2D(cv::Point2f(centre, centre), - idx * 15, 1.0);
            cv::Mat bK;
            cv::warpAffine(bottom_half_kernel, bK, topM, cv::Size(size, size));
            
            cv::Mat result_kernel = tK - bK;
            result_kernel = result_kernel - kernel / radiuses[r];
            
            oKs.push_back(result_kernel);
        }
        this->TGkernels.push_back(oKs);
    }
    
    // Get line_l5_kernels lien_l3_kernels line_ep_left_kernels line_ep_right_kernels
    for(int i = 0; i < this->numLine; i++)
    {
        line_l5_kernels.push_back(getLineKernel(5, i * 180 / this->numLine, 0, 50));
        line_l3_kernels.push_back(getLineKernel(3, i * 180 / this->numLine, 0, 50));
        line_ep_left_kernels.push_back(getLineKernel(3, i * 180 / this->numLine, 1, 50));
        line_ep_right_kernels.push_back(getLineKernel(3, i * 180 / this->numLine, 2, 50));
        line_ep_kernels.push_back(line_ep_left_kernels[i] - line_ep_right_kernels[i]);
    }
    

#if TEST_EDGEDETECTORHYPERPARAMS
    for(int i = 0; i < this->GaussianGradientKernels.size(); i++)
    {
        int delta_angle = 180 / numOrient;
        int thetaV = i * delta_angle;
        writeImage(visuKernel(this->GaussianGradientKernels[i]), "GaussianGradientKernel_" + to_string(i), true);
        cout << this->GaussianGradientKernels[i] << endl;
        cout << "sum: " << cv::sum(GaussianGradientKernels[i])[0] << endl;
    }
#endif
}
