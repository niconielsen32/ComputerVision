#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <string>

int main() {
  std::string path1 = "../src/khaled.jpg";
  std::string path2 = "../src/light.jpg";
  std::string path3 = "../src/light2.png";

  cv::Mat image1 = cv::imread(path1, cv::IMREAD_COLOR);
  cv::Mat image2 = cv::imread(path2, cv::IMREAD_COLOR);
  cv::Mat image3 = cv::imread(path3, cv::IMREAD_COLOR);
  if (image1.empty() || image2.empty() || image3.empty()) {
    std::cerr << "Could not open or find the image: " << std::endl;
    return 1;
  }
  cv::Mat gray1, gray2, gray3;
  cv::cvtColor(image1, gray1, CV_BGR2HSV);
  cv::cvtColor(image2, gray2, CV_BGR2HSV);
  cv::cvtColor(image3, gray3, CV_BGR2HSV);
  cv::Mat hsv_half =
      gray1(cv::Range(gray1.rows / 2, gray1.rows), cv::Range(0, gray1.cols));

  int h_bins = 50, s_bins = 60;
  int histSize[] = {h_bins, s_bins};
  float h_ranges[] = {0, 180};
  float s_ranges[] = {0, 256};
  const float *ranges[] = {h_ranges, s_ranges};
  int channels[] = {0, 1};

  cv::Mat hist_base, hist_half_down, hist_test1, hist_test2;
  calcHist(&gray1, 1, channels, cv::Mat(), hist_base, 2, histSize, ranges, true,
           false);
  normalize(hist_base, hist_base, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
  calcHist(&hsv_half, 1, channels, cv::Mat(), hist_half_down, 2, histSize,
           ranges, true, false);
  normalize(hist_half_down, hist_half_down, 0, 1, cv::NORM_MINMAX, -1,
            cv::Mat());
  calcHist(&gray2, 1, channels, cv::Mat(), hist_test1, 2, histSize, ranges,
           true, false);
  normalize(hist_test1, hist_test1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
  calcHist(&gray3, 1, channels, cv::Mat(), hist_test2, 2, histSize, ranges,
           true, false);
  normalize(hist_test2, hist_test2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
  for (int compare_method = 0; compare_method < 4; compare_method++) {
    double base_base = compareHist(hist_base, hist_base, compare_method);
    double base_half = compareHist(hist_base, hist_half_down, compare_method);
    double base_test1 = compareHist(hist_base, hist_test1, compare_method);
    double base_test2 = compareHist(hist_base, hist_test2, compare_method);
    std::cout << "Method " << compare_method
              << " Perfect, Base-Half, Base-Test(1), Base-Test(2) : "
              << base_base << " / " << base_half << " / " << base_test1 << " / "
              << base_test2 << std::endl;
  }
  std::cout << "Done \n";
  return 0;
}
