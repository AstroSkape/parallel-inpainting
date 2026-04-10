#include <cstddef>
#include <iostream>

#include "../include/inpaint.h"

using namespace std;

int main()
{
    std::cout << "Hello" << endl;
    // read image
    cv::Mat prunedImg = cv::imread("../images/forest_pruned.bmp", cv::IMREAD_COLOR);
    
    // create mask
    cv::Mat mask = cv::Mat(prunedImg.size(), CV_8UC1);
    for (size_t i = 0; i < prunedImg.size().height; i++)
    {
        for (size_t j = 0; j < prunedImg.size().width; j++)
        {
            cv::Vec3b pixel = prunedImg.at<cv::Vec3b>(i, j);
            if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
                mask.at<unsigned char>(i, j) = 1;
        }
    }

    auto metric = PatchSSDDistanceMetric(3);
    auto result = Inpainting(prunedImg, mask, &metric).run(true, false);

    // cv::imshow("Result", result);
    // cv::waitKey();

    bool success = cv::imwrite("../images/output.png", result);

    return 0;
}