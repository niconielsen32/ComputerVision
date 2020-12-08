#ifndef HOUGH_H
#define HOUGH_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

double lidarDistanceMarble = 0;

double realMarbleDistance = 4.241715;

vector<double> realDistMarble;
vector<double> estimatedDistMarble = {3.87942, 4.59784, 4.43363, 4.00457, 4.28074, 4.13805};

double angleToDetectedMarble = 0;

bool saveDistance = false;


void dataToCSV(){

    ofstream outputFile;
    string filename = "marbleDistance.csv";

    outputFile.open(filename);

    for(auto &realDist : realDistMarble){
        outputFile << realDist << ",";
    }

    outputFile << endl;

    for(auto &estimatedDist : estimatedDistMarble){
        outputFile << estimatedDist << ",";
    }

    outputFile.close();


    cout << "Data sent to CSV file." << endl;

}

double calculateAngleToMarble(std::vector<cv::Vec3f> circles, int closestCircleDetected){

    const double horizontalResolution = 320;
    const double widthFromMid = horizontalResolution/2;
    const double horizontalFOV = 60;
    const double hFOVmiddle = horizontalFOV/2;

    int xCoordinateCircle = circles[closestCircleDetected][0];

    double displacementRatio = double(xCoordinateCircle) / widthFromMid;

    double marbleAngleHorizontal = (displacementRatio - 1) * hFOVmiddle;

    return marbleAngleHorizontal;


    //std::cout << "Horizontal Angle: " << marbleAngleHorizontal << std::endl;


}


double calculateDistanceToMarble(const double &radius){

    const double pixelToMM = 0.2645833333;
    double distance, distanceToMarble;

   // const double horiRes = 320;
    const double horiResMM = 84.66;

   // const double widthFromMid = horiRes/2;
    const double FOVdegHalf = 30.0;
    double FOVrad = FOVdegHalf * CV_PI/180.0; // 1.0472 rad
   // const double hFOVmiddle = FOV/2;

    //cout << "FOV: " << FOVrad << endl;

    double focalLenght = 0.0;

    focalLenght = horiResMM / (2.0 * tan(FOVrad));

    //cout << "focal length: " << focalLenght << endl;

//    double xR = (center.x + radius) / focalLenght;
//    double yR = center.y / focalLenght;
//    double xL = (center.x - radius) / focalLenght;
//    double yL = center.y / focalLenght;


//    double distanceIn3D = sqrt(pow(xR - xL, 2.0) + pow(yR - yL, 2.0));

//    distanceIn3D *= pixelToMM;

//    cout << "dist 3d: " << distanceIn3D << endl;


    double diameterDetectedMarble = radius / 2.0;

    diameterDetectedMarble *= pixelToMM;

    //cout << "dist 2d: " << diameterDetectedMarble << endl;

    distance = (focalLenght * horiResMM) / diameterDetectedMarble;

    distanceToMarble = (distance * pixelToMM) / 100.0;


//    cout << "dist: " << distanceToMarble << endl;
//    cout << "Lidar dist: " << lidarDistanceMarble << endl;

    double percentageDeviation = abs((distanceToMarble - lidarDistanceMarble)) / lidarDistanceMarble * 100.0;

    //cout << "deviation: " << percentageDeviation << "%" << endl;

    cout << endl;

//    if(saveDistance){
//        realDistMarble.push_back(lidarDistanceMarble);
//        estimatedDistMarble.push_back(distanceToMarble);

//        saveDistance = false;
//    }

    return distanceToMarble;
}



double houghDetection(cv::Mat &gray, cv::Mat &imgOutput, int cannyValue, int accumulatorValue){

    double distToMarble;

    // Vector that contains the output from the hough transformation function
    std::vector<cv::Vec3f> circles;

    // opencv built in function to detect circles in frame
    HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows, cannyValue, accumulatorValue, 0, 0 );

    //Set if marble is detected
    bool marbleDetected = false;
    marbleDetected = !circles.empty();

    int closestCircleDetected = 0;

    cv::cvtColor(gray, imgOutput, cv::COLOR_GRAY2RGB);

    // Get radius and outline of circles detected - We are interested in the closest and largest circle
    if(marbleDetected == true)
    {
      for( size_t i = 0; i < circles.size(); i++ )
      {
          cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
          // center of the circle
          double radius = cvRound(circles[i][2]);

          distToMarble = calculateDistanceToMarble(radius);

          //cout << "dist to marble: " << distToMarble << endl;

          circle(imgOutput, center, 3, cv::Scalar(0,0,255), -1, 8, 0 );
          // outline of the cirle
          circle(imgOutput, center, radius, cv::Scalar(0,255,0), 3, 8, 0 );
          // Check if the circle is the closest
          if(radius > circles[closestCircleDetected][2])
          {
            closestCircleDetected = i;

          }
      }

    angleToDetectedMarble = calculateAngleToMarble(circles, closestCircleDetected);

    cout << "Angle to marble from center: " << angleToDetectedMarble << endl;

    }
    return distToMarble;
}






#endif // HOUGH_H
