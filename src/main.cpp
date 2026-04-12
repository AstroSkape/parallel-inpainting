#include <cstddef>
#include <iostream>

#include "../include/inpaint.h"
#include "../include/CycleTimer.h"

using namespace std;

int main()
{
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

    bool is_gpu_enabled = false;
    auto metric = PatchSSDDistanceMetric(3);
    double startTime = CycleTimer::currentSeconds();
    auto result = Inpainting(prunedImg, mask, &metric, is_gpu_enabled).run(true, false);
    double endTime = CycleTimer::currentSeconds();

    // cv::imshow("Result", result);
    // cv::waitKey();

    std::string ctx = is_gpu_enabled ? "GPU" : "CPU";
    printf("Processing Time (%s): %lfs\n", ctx.c_str(), endTime - startTime);
    bool success = cv::imwrite("../images/output.png", result);

    return 0;
}