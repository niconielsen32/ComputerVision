#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
 
using namespace std;
using namespace cv;
using namespace dnn;
 
 
int main(int, char**) {

    string file_path = "C:/Users/nhoei/ComputerVision/opencvGPU/";
    vector<string> class_names;
    ifstream ifs(string(file_path + "object_detection_classes_coco.txt").c_str());
    string line;

    // Load in all the classes from the file
    while (getline(ifs, line))
    {   
        cout << line << endl;
        class_names.push_back(line);
    } 
    

    // Read in the neural network from the files
    auto net = readNet(file_path + "frozen_inference_graph.pb",
    file_path + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt","TensorFlow");
 

    // Open up the webcam
    VideoCapture cap(0);
 

    // Run on either CPU or GPU
    //net.setPreferableBackend(DNN_BACKEND_CUDA);
    //net.setPreferableTarget(DNN_TARGET_CUDA);


    // Set a min confidence score for the detections
    float min_confidence_score = 0.5;


    // Loop running as long as webcam is open and "q" is not pressed
    while (cap.isOpened()) {

        // Load in an image
        Mat image;
        bool isSuccess = cap.read(image);

        // Check if image is loaded in correctly
        if (!isSuccess){
            cout << "Could not load the image!" << endl;
            break;
        }
        
        int image_height = image.cols;
        int image_width = image.rows;



        auto start = getTickCount();

        // Create a blob from the image
        Mat blob = blobFromImage(image, 1.0, Size(300, 300), Scalar(127.5, 127.5, 127.5),
                                true, false);

        
        // Set the blob to be input to the neural network
        net.setInput(blob);

        // Forward pass of the blob through the neural network to get the predictions
        Mat output = net.forward();

        auto end = getTickCount();
        


        // Matrix with all the detections
        Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());
        
        // Run through all the predictions
        for (int i = 0; i < results.rows; i++){
            int class_id = int(results.at<float>(i, 1));
            float confidence = results.at<float>(i, 2);
    
            // Check if the detection is over the min threshold and then draw bbox
            if (confidence > min_confidence_score){
                int bboxX = int(results.at<float>(i, 3) * image.cols);
                int bboxY = int(results.at<float>(i, 4) * image.rows);
                int bboxWidth = int(results.at<float>(i, 5) * image.cols - bboxX);
                int bboxHeight = int(results.at<float>(i, 6) * image.rows - bboxY);
                rectangle(image, Point(bboxX, bboxY), Point(bboxX + bboxWidth, bboxY + bboxHeight), Scalar(0,0,255), 2);
                string class_name = class_names[class_id-1];
                putText(image, class_name + " " + to_string(int(confidence*100)) + "%", Point(bboxX, bboxY - 10), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0,255,0), 2);
            }
        }
        

        auto totalTime = (end - start) / getTickFrequency();
        

        putText(image, "FPS: " + to_string(int(1 / totalTime)), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 255, 0), 2, false);
        
        imshow("image", image);


        int k = waitKey(10);
        if (k == 113){
            break;
        }
    }
    
    cap.release();
    destroyAllWindows();
}