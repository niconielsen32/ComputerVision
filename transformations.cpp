#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "mouse.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    // Read image from file 
    Mat snookerImg = imread("C:/Users/kemik/OneDrive/Dokumenter/Downloads/snooker.jpg");
    Mat plateImg = imread("C:/Users/kemik/OneDrive/Dokumenter/Downloads/plate.jpg");
    Mat perspecImg, affineImg;

    //if fail to read the image
    if (snookerImg.empty() || plateImg.empty())
    {
        cout << "Error loading the image" << endl;
        return -1;
    }

    //Create a window
    namedWindow("Original Plate", 1);
    namedWindow("Original Snooker", 1);

    // Find points in image with mouse
    setMouseCallback("Original Plate", platePoints, NULL);
    setMouseCallback("Original Snooker", snookerPoints, NULL);

    imshow("Original Plate", plateImg);
    imshow("Original Snooker", snookerImg);

    waitKey(0);

    cout << "Points Plate: " << endl;
    for (auto& i : pointsPlate) {
        cout << "( " << i.x << ", " << i.y << " )" << endl;
    }

    cout << "Destination Plate: " << endl;
    for (auto& i : destinationPlate) {
        cout << "( " << i.x << ", " << i.y << " )" << endl;
    }


    // Affine Transforamtion of number plate
    // Find points from image and distination points
    //vector<Point2f> sourcePlate, destinationPlate;
    //sourcePlate = { Point2f(400,250), Point2f(400,320), Point2f(200,330) };
    //destinationPlate = { Point2f(770,350), Point2f(770,450), Point2f(250,370) };


    // Calculate the affine matrix from the found points in the image
    Mat affineMatrix = getAffineTransform(pointsPlate, destinationPlate);
    // Apply the affine transformation on the image
    warpAffine(plateImg, affineImg, affineMatrix, plateImg.size());

    //show the image
    imshow("Plate Transformation", affineImg);



    // Perspective Transformation of snooker table
    // Find points from image and distination points
    //vector<Point2f> sourceSnooker, destinationSnooker;
    //sourceSnooker = { Point2f(338,645), Point2f(671,650), Point2f(922,916), Point2f(101,919) };
    //destinationSnooker = { Point2f(278,223), Point2f(785,220), Point2f(830,905), Point2f(205,907) };


    // Calculate the perspective matrix from the found points in the image
    Mat perspectiveMatrix = getPerspectiveTransform(pointsSnooker, destinationSnooker);
    // Apply the perspective transformatin on the image
    warpPerspective(snookerImg, perspecImg, perspectiveMatrix, snookerImg.size());

    // Show image
    imshow("Perspective Transformation", perspecImg);


    // Wait until user press some key
    waitKey(0);

    return 0;

}
