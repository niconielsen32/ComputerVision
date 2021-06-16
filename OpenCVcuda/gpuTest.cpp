#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>


using namespace std;
using namespace cv;


// Open up the webcam
VideoCapture cap(0);


void cpuSpeedTest() {


    while (cap.isOpened()) {

        Mat image;
        bool isSuccess = cap.read(image);

        if (image.empty()) {
            cout << "Could not load in image!" << endl;
        }

        auto start = getTickCount();

        Mat result;

        bilateralFilter(image, result, 50, 100, 100);

        auto end = getTickCount();

        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1 / totalTime;
        cout << "FPS: " << fps << endl;

        putText(result, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);

        imshow("Image", result);

        int k = waitKey(10);
        if (k == 113) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

}




void gpuSpeedTest() {


    while (cap.isOpened()) {

        Mat image;
        bool isSuccess = cap.read(image);

        cuda::GpuMat imgGPU;

        imgGPU.upload(image);


        if (imgGPU.empty()) {
            cout << "Could not load in image on GPU!" << endl;
        }


        auto start = getTickCount();

        cuda::bilateralFilter(imgGPU, imgGPU, 50, 100, 100);

        auto end = getTickCount();

        auto totalTime = (end - start) / getTickFrequency();
        auto fps = 1 / totalTime;
        cout << "FPS: " << fps << endl;

        imgGPU.download(image);

        putText(image, "FPS: " + to_string(int(fps)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
        imshow("Image", image);

        int k = waitKey(10);
        if (k == 113) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();

}


int main(int, char**) {


    gpuSpeedTest();

    return 0;
}
