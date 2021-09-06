#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafeatures2d.hpp>

using namespace std;
using namespace cv;

VideoCapture cap(0);

RNG rng(12345);//random number

int main()
{   

    //cuda::printCudaDeviceInfo(0);

    Mat img;
    cuda::GpuMat imgGpu, gray, circlesGpu;

    vector<Vec3f> circles;


    while(cap.isOpened()){

        auto start = getTickCount();

        cap.read(img);
      
        imgGpu.upload(img);

        cuda::cvtColor(imgGpu, gray, COLOR_BGR2GRAY);

        // Image Filtering
        auto gaussianFilter = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, {3,3}, 1);
        gaussianFilter->apply(gray, gray);

        // Circle Detector
        auto circleDetection = cuda::createHoughCirclesDetector(1,100, 120, 50, 1, 50, 5);
        circleDetection->detect(gray, circlesGpu);

        circles.resize(circlesGpu.size().width);
        if(!circles.empty()){
            circlesGpu.row(0).download(Mat(circles).reshape(3,1));
        }
        
       

        auto end = getTickCount();
        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1 / totalTime;

        for( size_t i = 0; i < circles.size(); i++ )
        {
            int b = rng.uniform(0,255);
            int g = rng.uniform(0,255);
            int r = rng.uniform(0,255);

            Vec3i cir = circles[i];
            circle(img, Point(cir[0], cir[1]), cir[2], Scalar(b,g,r), 2, LINE_AA);
        }

        putText(img, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2, false);
        imshow("Image", img);

        

        if(waitKey(1) == 'q'){
            break;
        }
    }

    
    cap.release();
    destroyAllWindows();
    
    return 0;
}
