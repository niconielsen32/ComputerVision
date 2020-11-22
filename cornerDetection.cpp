#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void harrisCornerDetector() {

    Mat image, gray;
    Mat output, output_norm, output_norm_scaled;

    // Loading the actual image
    //image = imread("C:/Users/kemik/OneDrive/Dokumenter/Downloads/squares.png", IMREAD_COLOR);
    image = imread("C:/Users/kemik/OneDrive/Dokumenter/Downloads/house.jpg", IMREAD_COLOR);

    // Edge cases
    if (image.empty()) {
        cout << "Error loading image" << endl;
    }
    cv::imshow("Original image", image);

    // Converting the color image into grayscale
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detecting corners using the cornerHarris built in function
    output = Mat::zeros(image.size(), CV_32FC1);
    cornerHarris(gray, output,
        3,              // Neighborhood size
        3,              // Aperture parameter for the Sobel operator
        0.04);          // Harris detector free parameter

    // Normalizing - Convert corner values to integer so they can be drawn
    normalize(output, output_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(output_norm, output_norm_scaled);

    // Drawing a circle around corners
    for (int j = 0; j < output_norm.rows; j++) {
        for (int i = 0; i < output_norm.cols; i++) {
            if ((int)output_norm.at<float>(j, i) > 100) {
                circle(image, Point(i, j), 4, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }

    // Displaying the result
    cv::resize(image, image, cv::Size(), 1.5, 1.5);
    cv::imshow("Output Harris", image);
    cv::waitKey();   
}




void shiTomasiCornerDetector() {

    Mat image, gray;
    Mat output, output_norm, output_norm_scaled;

    // Loading the actual image
    //image = imread("C:/Users/kemik/OneDrive/Dokumenter/Downloads/squares.png", IMREAD_COLOR);
    image = imread("C:/Users/kemik/OneDrive/Dokumenter/Downloads/house.jpg", IMREAD_COLOR);

    // Edge cases
    if (image.empty()) {
        cout << "Error loading image" << endl;
    }
    cv::imshow("Original image", image);

    // Converting the color image into grayscale
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);


    // Detecting corners using the goodFeaturesToTrack built in function
    vector<Point2f> corners;
    goodFeaturesToTrack(gray, 
                        corners,
                        100,            // Max corners to detect
                        0.01,           // Minimal quality of corners
                        10,             // Minimum Euclidean distance between the returned corners
                        Mat(),          // Optional region of interest
                        3,              // Size of an average block for computing a derivative covariation matrix over each pixel neighbothood
                        false,          // Use Harri Detector or cornerMinEigenVal - Like when you create your own
                        0.04);          // Free parameter for the Harris detector


    // Drawing a circle around corners
    for (size_t i = 0; i < corners.size(); i++){
        circle(image, corners[i], 4, Scalar(0, 255, 0), 2, 8, 0);
    }

    // Displaying the result
    cv::resize(image, image, cv::Size(), 1.5, 1.5);
    cv::imshow("Output Shi-Tomasi", image);
    cv::waitKey();
}


int main()
{

    harrisCornerDetector();
    shiTomasiCornerDetector();

    return 0;
}
