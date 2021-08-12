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

int main(int argc, char** argv )
{   

    cuda::printCudaDeviceInfo(0);

    Mat img;
    cuda::GpuMat imgGpu, gray;

    Mat corners;
    cuda::GpuMat cornersGpu;

    
    while(cap.isOpened()){

        auto start = getTickCount();

        cap.read(img);
        imgGpu.upload(img);

        cuda::cvtColor(imgGpu, gray, COLOR_BGR2GRAY);

        auto cornerDetector = cuda::createGoodFeaturesToTrackDetector(gray.type(), 100, 0.01, 10, 3, false);
        cornerDetector->detect(gray, cornersGpu);

        cornersGpu.download(corners);

        auto end = getTickCount();
        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1 / totalTime;

        for(int i = 0; i < corners.cols; i++){

            int b = rng.uniform(0,255);
            int g = rng.uniform(0,255);
            int r = rng.uniform(0,255);
            Point2f point = corners.at<Point2f>(0, i);
            circle(img, point, 6, Scalar(b,g,r), 2, 8);
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
