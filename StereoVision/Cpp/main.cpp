#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

#include "StereoVision.h"

using namespace cv;
using namespace std;


float baseline = 9.0;
float focalLenght = 6.0;
float alpha = 56.6;


int main() {

    Mat leftFrame, rightFrame;
    VideoCapture capLeft(0);
    VideoCapture capRight(1);

    StereoVision stereovision(baseline, alpha, focalLenght);

    if (!capLeft.isOpened()) {
        cout << "Cannot Open Left Camera" << endl;
        return -1;
    }

    if (!capRight.isOpened()) {
        cout << "Cannot Open Right Camera" << endl;
        return -1;
    }

    Mat leftMask, rightMask;
    Mat leftResFrame, rightResFrame;

    Point leftCircle, rightCircle;

    float ballDepth = 0;

    while (true) {

        capLeft.read(leftFrame);
        capRight.read(rightFrame);

        // Calibration of the frames
        //stereovision.undistortFrame(leftFrame);
        //stereovision.undistortFrame(rightFrame);
        
        
        // Applying HSV-filter
        leftMask = stereovision.add_HSV_filter(leftFrame, 0);
        rightMask = stereovision.add_HSV_filter(rightFrame, 1);

        
        // Frames after applyting HSV-filter mask
        bitwise_and(leftFrame, leftFrame, leftResFrame, leftMask);
        bitwise_and(rightFrame, rightFrame, rightResFrame, rightMask);

        
        // Detect Circles - Hough Transforms can be used aswell or some neural network to do object detection
        leftCircle = stereovision.find_ball(leftFrame, leftMask);
        rightCircle = stereovision.find_ball(rightFrame, rightMask);

        

        // Calculate the depth of the ball

       // If no ball is detected in one of the cameras - show the text "tracking lost"
        if (!leftCircle.x || !rightCircle.x) {
            putText(leftFrame, "Tracking Lost", { 75, 50 }, FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
            putText(rightFrame, "Tracking Lost!", { 75, 75 }, FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2);
        } else {

            // Vector of all depths in case of several balls detected.
            // All formulas used to find depth is in the video presentation
            ballDepth = stereovision.find_depth(leftCircle, rightCircle, leftFrame, rightFrame);

            putText(leftFrame, "Tracking!", { 75, 50 }, FONT_HERSHEY_SIMPLEX, 0.7, (125, 250, 0), 2);
            putText(rightFrame, "Tracking!", { 75, 75 }, FONT_HERSHEY_SIMPLEX, 0.7, (125, 250, 0), 2);


            // Multiply computer value with 205.8 to get real - life depth in[cm]. The factor was found manually.
            cout << "Ball depth: " << ballDepth << endl;
 
        }

        // Show the frames
        imshow("Left Frame", leftFrame);
        imshow("Right Frame", rightFrame);
        imshow("Left Mask", leftMask);
        imshow("Right Mask", rightMask);

        // Hit "q" to close the window
        if ((waitKey(1) & 0xFF) == 'q') {
            break;
        }
    }

    // Release and destroy all windows before termination
    capLeft.release();
    capRight.release();

    destroyAllWindows();

    return 0;
}
