#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat img, gray, blurred, edge;

// Laplacian Operator Variables
int kernel_size = 3;
int ddepth = CV_16S;

// Canny Edge Detection Variables
int lowerThreshold = 0;
int max_lowThreshold = 100;


void laplacianDetection() {

    GaussianBlur(gray,
        blurred,
        cv::Size(3, 3),  // smoothing window width and height in pixels
        3);              // how much the image will be blurred

    Laplacian(blurred,
        edge,
        ddepth,         // Depth of the destination image
        kernel_size);    // Size of the kernel

    // converting back to CV_8U
    convertScaleAbs(edge, edge);
}


void CannyThreshold(int, void*) {

    GaussianBlur(gray,
        blurred,
        cv::Size(3, 3),  // smoothing window width and height in pixels
        3);              // how much the image will be blurred

    Canny(blurred,
        edge,
        lowerThreshold, // lower threshold
        50);           // higher threshold

    imshow("Edge Detection", edge);
}



int main() {

    img = imread("/Users/kemik/OneDrive/Skrivebord/lenna.png");

    if (img.empty())
    {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    cvtColor(img, gray, COLOR_BGR2GRAY);

    cv::namedWindow("Original", WINDOW_AUTOSIZE);
    cv::namedWindow("Gray", WINDOW_AUTOSIZE);
    cv::namedWindow("Blurred", WINDOW_AUTOSIZE);
    cv::namedWindow("Edge Detection", WINDOW_AUTOSIZE);


    // Canny Edge Detector
    createTrackbar("Min Threshold:", "Edge Detection", &lowerThreshold, max_lowThreshold, CannyThreshold);
    CannyThreshold(0,0);

    // Laplacian Edge Detector
    //laplacianDetection();

    imshow("Original", img);
    imshow("Gray", gray);
    imshow("Blurred", blurred);
    //imshow("Edge Detection", edge);

    waitKey(0);
  
    

  return 0;
}
