#pragma once
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

vector<Point2f> pointsPlate;
vector<Point2f> destinationPlate;
vector<Point2f> pointsSnooker;
vector<Point2f> destinationSnooker;

void platePoints(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN and pointsPlate.size() < 3)
    {
        cout << "Plate - position (" << x << ", " << y << ")" << endl;
        pointsPlate.push_back(Point2f(x, y));
    }
    else if (event == EVENT_LBUTTONDOWN) {
        cout << "Plate - destination (" << x << ", " << y << ")" << endl;
        destinationPlate.push_back(Point2f(x, y));
    }
}

void snookerPoints(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN and pointsSnooker.size() < 4)
    {
        cout << "Snooker - position (" << x << ", " << y << ")" << endl;
        pointsSnooker.push_back(Point2f(x, y));
    }
    else if (event == EVENT_LBUTTONDOWN) {
        cout << "Snooker - destination (" << x << ", " << y << ")" << endl;
        destinationSnooker.push_back(Point2f(x, y));
    }
}

