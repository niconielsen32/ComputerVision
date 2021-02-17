#include <iostream>
#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Ptr<BackgroundSubtractor> pBackSub;
    pBackSub = createBackgroundSubtractorKNN();

    // Start default camera
    VideoCapture video(0);

    double fps = video.get(CAP_PROP_FPS);
    cout << "Frames per second camera : " << fps << endl;

    // Number of frames to capture
    int num_frames = 1;

    // Start and end times
    clock_t start;
    clock_t end;

    Mat frame, fgMask;

    cout << "Capturing " << num_frames << " frames" << endl;

    double ms, fpsLive;

    //chrono::steady_clock::time_point start, end;
    //std::common_type_t<std::chrono::steady_clock::duration, std::chrono::steady_clock::duration> diff;

    int keyboard;

    while(true){


        // Start time
        start = clock();
        //start = chrono::steady_clock::now();

        video >> frame;

        pBackSub->apply(frame, fgMask);

        erode(fgMask, fgMask, (5, 5));
        dilate(fgMask, fgMask, (5, 5));

        long sum = 0;
        int N = 1;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                sum += 1;
            }
        }

        // End Time
        end = clock();
        //end = chrono::steady_clock::now();

        //diff = end - start;

        //cout << "start: " << start << endl;
        //cout << "end: " << end << endl;

        // Time elapsed
        double seconds =  (double(end) - double(start)) / double(CLOCKS_PER_SEC);
        cout << "Time taken : " << seconds << " seconds" << endl;
        //ms = chrono::duration <double, milli>(diff).count();
        //cout << ms << " ms" << endl;

        // Calculate frames per second
        fpsLive = double(num_frames) / double(seconds);
        //fpsLive = double(num_frames) / double(ms*0.001);
        cout << "Estimated frames per second : " << fpsLive << endl;

        putText(fgMask, "FPS: " + to_string(int(fpsLive)), { 50, 50 }, FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),2);

        imshow("fgMask", fgMask);

        keyboard = waitKey(30);
        if (keyboard == 'q' or keyboard == 27) {
            break;
        }
    }


    // Release video
    video.release();
    return 0;
}
