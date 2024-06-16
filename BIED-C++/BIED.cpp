#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "PathManager.h"
#include "EdgeDetector.h"

using namespace std;

int main(int argc, char **argv)
{
    string imgsDir = "BSDS2"; // the directory of images
    PathManager pm(imgsDir);
    EdgeDetector ed;
    clock_t runningStart = clock();
    while(pm.beingCurFile)
    {
        // clock_t start = clock();
        string imgPath = pm.getCurImgPath();
        cv::Mat image = cv::imread(imgPath);
        cout << "Now process: " << pm.getFilename() << endl;

        // Start to detect edge
        ed.BIED(pm.getFilename(), image);
        // cout << "The edge detection cost: " << double(clock() - start) / CLOCKS_PER_SEC << " s" << endl;

        // Get the information of the next image
        pm.nextFile();
        
        // save
        writeImage(ed.edge, ed.imageName, true, "edge");
    }
    cout << "The average running time is " << double(clock() - runningStart) / CLOCKS_PER_SEC / pm.getImagesNumber()  << " s" << endl;
}
