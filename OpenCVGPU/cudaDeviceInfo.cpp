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
