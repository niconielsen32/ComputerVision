#include <gazebo/gazebo_client.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <math.h>


static boost::mutex mutexx;

using namespace std;

//double lidarDistanceMarble = 0;

#include "hough.h"


void statCallback(ConstWorldStatisticsPtr &_msg) {
  (void)_msg;
  // Dump the message contents to stdout.
  //  std::cout << _msg->DebugString();
  //  std::cout << std::flush;
}

void poseCallback(ConstPosesStampedPtr &_msg) {
  // Dump the message contents to stdout.
  //  std::cout << _msg->DebugString();

  for (int i = 0; i < _msg->pose_size(); i++) {
    //if (_msg->pose(i).name() == "pioneer2dx") {

//      std::cout << std::setprecision(2) << std::fixed << std::setw(6)
//                << _msg->pose(i).position().x() << std::setw(6)
//                << _msg->pose(i).position().y() << std::setw(6)
//                << _msg->pose(i).position().z() << std::setw(6)
//                << _msg->pose(i).orientation().w() << std::setw(6)
//                << _msg->pose(i).orientation().x() << std::setw(6)
//                << _msg->pose(i).orientation().y() << std::setw(6)
//                << _msg->pose(i).orientation().z() << std::endl;
    //}

//    if(_msg->pose(i).name() == "marble"){
//       std::cout << std::setprecision(2) << std::fixed << std::setw(6)
//            << _msg->pose(i).position().x() << std::setw(6)
//            << _msg->pose(i).position().y() << std::setw(6)
//            << _msg->pose(i).position().z() << std::setw(6)  << std::endl;
//    }
  }
}


void cameraCallback(ConstImageStampedPtr &msg) {

  std::size_t width = msg->image().width();
  std::size_t height = msg->image().height();
  const char *data = msg->image().data().c_str();
  cv::Mat im(int(height), int(width), CV_8UC3, const_cast<char *>(data));

  im = im.clone();
  cv::cvtColor(im, im, cv::COLOR_RGB2GRAY);

  // Reduce the noise to reduce/remove false detections
  //cv::medianBlur(img, img, 9);
  cv::GaussianBlur(im, im, cv::Size(9, 9), 2, 2 );
  //cv::fastNlMeansDenoising(img, img, 3);


  // Detect the marbles
  double distToMarble = houghDetection(im, im, 13, 19);

  if(distToMarble != 0){
  cv::putText(im, "Distance: " + std::to_string(distToMarble),
              cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(255, 0, 0));

  }

  cv::putText(im, "Angle: " + std::to_string(angleToDetectedMarble),
              cv::Point(10, 35), cv::FONT_HERSHEY_PLAIN, 1.0,
              cv::Scalar(255, 0, 0));




  mutexx.lock();
  cv::imshow("camera", im);
  mutexx.unlock();

}

void lidarCallback(ConstLaserScanStampedPtr &msg) {

  //  std::cout << ">> " << msg->DebugString() << std::endl;
  float angle_min = float(msg->scan().angle_min());
  //  double angle_max = msg->scan().angle_max();
  float angle_increment = float(msg->scan().angle_step());

  float range_min = float(msg->scan().range_min());
  float range_max = float(msg->scan().range_max());

  int sec = msg->time().sec();
  int nsec = msg->time().nsec();

  int nranges = msg->scan().ranges_size();
  int nintensities = msg->scan().intensities_size();

  assert(nranges == nintensities);

  int width = 400;
  int height = 400;
  float px_per_m = 200 / range_max;

  cv::Mat im(height, width, CV_8UC3);
  im.setTo(0);
  for (int i = 0; i < nranges; i++) {
    float angle = angle_min + i * angle_increment;
    float range = std::min(float(msg->scan().ranges(i)), range_max);
    //    double intensity = msg->scan().intensities(i);
    cv::Point2f startpt(200.5f + range_min * px_per_m * std::cos(angle),
                        200.5f - range_min * px_per_m * std::sin(angle));
    cv::Point2f endpt(200.5f + range * px_per_m * std::cos(angle),
                      200.5f - range * px_per_m * std::sin(angle));
    //cv::line(im, startpt * 16, endpt * 16, cv::Scalar(255, 255, 255, 255), 1,
           //  cv::LINE_AA, 4);

    //    std::cout << angle << " " << range << " " << intensity << std::endl;
  }
  cv::circle(im, cv::Point(200, 200), 2, cv::Scalar(0, 0, 255));
//  cv::putText(im, std::to_string(sec) + ":" + std::to_string(nsec),
//              cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1.0,
//              cv::Scalar(255, 0, 0));


  std::vector<float> frontRanges = {};
  std::vector<float> rangesTest = {};

  const int rightRange = 95, frontRange = 105, leftRange = 200-30;

  //RIGHT LIDAR RANGE
  // go through all the lidar points - get distance - draw points and lines for displaying lidar data
  for (int i = 30; i < rightRange; i++) {
  // angle for the lidar point
      float angle = angle_min + i * angle_increment;
      // distance from robot to where the lidar point hit an obstacle
      float range = std::min(float(msg->scan().ranges(i)), range_max);
      //    double intensity = msg->scan().intensities(i);
      //rightRanges.push_back(range);
      rangesTest.push_back(range);

          cv::Point2f startpt(200.5f + range_min * px_per_m * std::cos(angle),
                              200.5f - range_min * px_per_m * std::sin(angle));
          cv::Point2f endpt(200.5f + range * px_per_m * std::cos(angle),
                            200.5f - range * px_per_m * std::sin(angle));
          cv::line(im, startpt * 16, endpt * 16, cv::Scalar(255, 0, 0, 255), 1,
                   cv::LINE_AA, 4);

  }


  //std::vector<float> rangesTest;
  // FRONT LIDAR RANGE
  for (int i = rightRange; i < frontRange; i++) {
      // angle for the lidar point
      float angle = angle_min + i * angle_increment;
      // distance from robot to where the lidar point hit an obstacle
      float range = std::min(float(msg->scan().ranges(i)), range_max);

      frontRanges.push_back(range);
      rangesTest.push_back(range);


      //    double intensity = msg->scan().intensities(i);
      cv::Point2f startpt(200.5f + range_min * px_per_m * std::cos(angle),
                          200.5f - range_min * px_per_m * std::sin(angle));
      cv::Point2f endpt(200.5f + range * px_per_m * std::cos(angle),
                        200.5f - range * px_per_m * std::sin(angle));
      cv::line(im, startpt * 16, endpt * 16, cv::Scalar(0, 0, 255, 255), 1,
               cv::LINE_AA, 4);

  }

  //LEFT LIDAR RANGE
  for (int i = frontRange; i < leftRange; i++) {
      // angle for the lidar point
      float angle = angle_min + i * angle_increment;
      // distance from robot to where the lidar point hit an obstacle
      float range = std::min(float(msg->scan().ranges(i)), range_max);
      //    double intensity = msg->scan().intensities(i);

      //leftRanges.push_back(range);
      rangesTest.push_back(range);

          cv::Point2f startpt(200.5f + range_min * px_per_m * std::cos(angle),
                              200.5f - range_min * px_per_m * std::sin(angle));
          cv::Point2f endpt(200.5f + range * px_per_m * std::cos(angle),
                            200.5f - range * px_per_m * std::sin(angle));
          cv::line(im, startpt * 16, endpt * 16, cv::Scalar(0, 255, 0, 255), 1,
                   cv::LINE_AA, 4);

  }


  lidarDistanceMarble = *min_element(frontRanges.begin(), frontRanges.end());

  mutexx.lock();
  cv::imshow("lidar", im);
  mutexx.unlock();


}

int main(int _argc, char **_argv) {
  // Load gazebo
  gazebo::client::setup(_argc, _argv);

  // Create our node for communication
  gazebo::transport::NodePtr node(new gazebo::transport::Node());
  node->Init();

  // Listen to Gazebo topics
  gazebo::transport::SubscriberPtr statSubscriber =
      node->Subscribe("~/world_stats", statCallback);

  gazebo::transport::SubscriberPtr poseSubscriber =
      node->Subscribe("~/pose/info", poseCallback);

  gazebo::transport::SubscriberPtr cameraSubscriber =
      node->Subscribe("~/pioneer2dx/camera/link/camera/image", cameraCallback);

  gazebo::transport::SubscriberPtr lidarSubscriber =
      node->Subscribe("~/pioneer2dx/hokuyo/link/laser/scan", lidarCallback);

  // Publish to the robot vel_cmd topic
  gazebo::transport::PublisherPtr movementPublisher =
      node->Advertise<gazebo::msgs::Pose>("~/pioneer2dx/vel_cmd");

  // Publish a reset of the world
  gazebo::transport::PublisherPtr worldPublisher =
      node->Advertise<gazebo::msgs::WorldControl>("~/world_control");
  gazebo::msgs::WorldControl controlMessage;
  controlMessage.mutable_reset()->set_all(true);
  worldPublisher->WaitForConnection();
  worldPublisher->Publish(controlMessage);

  const int key_left = 81;
  const int key_up = 82;
  const int key_down = 84;
  const int key_right = 83;
  const int key_esc = 27;

  float speed = 0.0;
  float dir = 0.0;


  cv::Mat map;
  map = cv::imread("floor_plan.png");

  int scaleFactor = 20;

  cv::resize(map, map, cv::Size(map.cols * scaleFactor, map.rows * scaleFactor), 0, 0, cv::INTER_NEAREST);

  cv::Point2f robotPoint(map.cols/2, map.rows/2);

  cv::Point2f marblePoint((realMarbleDistance + 2)*scaleFactor + map.cols/2, map.rows/2);

  for(auto &dist : estimatedDistMarble){

      cv::Point2f distPoint((dist + 2)*scaleFactor + map.cols/2, map.rows/2);

      cv::circle(map, distPoint, 8, cv::Scalar(0,255,0));
  }

  cv::circle(map, marblePoint, 10, cv::Scalar(0, 0, 255));

  cv::circle(map, robotPoint, 8, cv::Scalar(255,0,0));



  // Loop
  while (true) {
    gazebo::common::Time::MSleep(10);

    mutexx.lock();
    int key = cv::waitKey(1);
    mutexx.unlock();


    cv::imshow("Map", map);


    if (key == key_esc)
      break;

    if ((key == key_up) && (speed <= 1.2f))
      speed += 0.05;
    else if ((key == key_down) && (speed >= -1.2f))
      speed -= 0.05;
    else if ((key == key_right) && (dir <= 0.4f))
      dir += 0.05;
    else if ((key == key_left) && (dir >= -0.4f))
      dir -= 0.05;
    else {
      // slow down
      //      speed *= 0.1;
      //      dir *= 0.1;
    }

    if( key == 's'){
        speed = 0.0;
    }

    if(key == 'o'){
        saveDistance = true;
    }


    if(key == 'p'){
        dataToCSV();
    }

    // Generate a pose
    ignition::math::Pose3d pose(double(speed), 0, 0, 0, 0, double(dir));

    // Convert to a pose message
    gazebo::msgs::Pose msg;
    gazebo::msgs::Set(&msg, pose);
    movementPublisher->Publish(msg);
  }


  // Make sure to shut everything down.
  gazebo::client::shutdown();
}
