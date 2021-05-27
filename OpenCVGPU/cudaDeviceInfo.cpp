// REMEMBER TO ADD THE OPENCV BIN FOLDER TO THE PATH IN ENVIRONMENTAL VARIABLES
// C:\your_path\opencv\build\install\x64\vc16\bin

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace cuda;

int main(int, char**) {

    printCudaDeviceInfo(0);

    cout << "Hello, world!\n";
}
