//Opencv C++ Example on Real time Face Detection Using Haar Cascade Classifier

/*We can similarly train our own Haar Classifier and Detect any object which we want
Only Thing is we need to load our Classifier in place of cascade_frontalface_alt2.xml */

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

void faceDetection(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

    // Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 6);
        Mat faceROI = frame_gray(faces[i]);
    }

    imshow("Live Face Detection", frame);
}


int main(int argc, const char** argv)
{

    // Load the pre trained haar cascade classifier

    //string faceClassifier = "C:/Users/kemik/OneDrive/Skrivebord/haarcascade_frontalface_default.xml";
    string faceClassifier = "C:/Users/kemik/OneDrive/Skrivebord/haarcascade_frontalface_alt2.xml";

    if (!face_cascade.load(faceClassifier))
    {
        cout << "Could not load the classifier";
        return -1;
    };

    cout << "Classifier Loaded!" << endl;

    // Read the video stream from camera
    VideoCapture capture(0);

    if (!capture.isOpened())
    {
        cout << "Could not open video capture";
        return -1;
    }

    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "No frame captured from camera";
            break;
        }

        // Apply the face detection with the haar cascade classifier
        faceDetection(frame);

        if (waitKey(10) == 'q')
        {
            break; // Terminate program if q pressed
        }
    }

    return 0;
}


