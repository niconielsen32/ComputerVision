#include "StereoVision.h"


void StereoVision::undistortFrame(Mat& frame) {

	Mat cameraMatrix, newCameraMatrix;
	vector<double> distortionParameters;

	newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distortionParameters, { frame.cols, frame.rows }, 1);

	undistort(frame, frame, cameraMatrix, distortionParameters, newCameraMatrix);
	

	// Can be calibrated as in the other video on the channel

	/*// Precompute lens correction interpolation
	Mat mapX, mapY;
	undistort(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1,
		mapX, mapY);

	// Show lens corrected images
	std::cout << std::string(f) << std::endl;

	cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);

	cv::Mat imgUndistorted;
	// 5. Remap the image using the precomputed interpolation maps.
	cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);*/

}



Mat StereoVision::add_HSV_filter(Mat& frame) {

	// Blurring the frame to reduce noise
	GaussianBlur(frame, frame, { 5,5 }, 0);

	// Convert to HSV
	cvtColor(frame, frame, COLOR_BGR2HSV);

	vector<int> lowerLimitRed = { 143,112,53 };     // Lower limit for red ball
	vector<int> upperLimitRed = { 255,255,255 };	 // Upper limit for red ball


	/*vector<int> lowerLimitBlue = { 140,106,0 };     // Lower limit for blue ball
	vector<int> upperLimitBlue = { 255,255,255 };	 // Upper limit for blue ball
	*/

	Mat mask;

	inRange(frame, lowerLimitRed, upperLimitRed, mask);

	erode(mask, mask, (3, 3));
	dilate(mask, mask, (3, 3));

	return mask;
}



Point StereoVision::find_ball(Mat& frame, Mat& mask) {

	vector<vector< Point> > contours;

	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Sort the contours to find the biggest one
	sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
		return contourArea(c1, false) < contourArea(c2, false);
	});

	if (contours.size() > 0) {
		vector<Point> largestContour = contours[contours.size() - 1];
		Point2f center;
		float radius;
		minEnclosingCircle(largestContour, center, radius);
		Moments m = moments(largestContour);
		Point centerPoint(m.m10 / m.m00, m.m01 / m.m00);

		// Only preceed if the radius is grater than a minimum threshold
		if (radius > 10) {
			// Draw the circle and centroid on the frame
			circle(frame, center, int(radius), (0, 255, 255), 2);
			circle(frame, centerPoint, 5, (0, 0, 255), -1);
		}

		return centerPoint;
	}
}



float StereoVision::find_depth(Point circleLeft, Point circleRight, Mat& leftFrame, Mat& rightFrame) {
	
	int focal_pixels = 0;

	if (rightFrame.cols == leftFrame.cols) {

		// Convert focal lenght f from [mm] to [pixel]
		focal_pixels = (rightFrame.cols * 0.5) / tan(alpha * 0.5 * CV_PI / 180.0);
	}
	else {
		cout << "Left and Right Camera frames do not have the same pixel width" << endl;
	}

	int xLeft = circleLeft.x;
	int xRight = circleRight.x;

	// Calculate the disparity
	int disparity = xLeft - xRight;

	// Calculate the Depth Z
	float zDepth = (baseline * float(focal_pixels)) / float(disparity);    // Depth in [cm]

	return zDepth;

}