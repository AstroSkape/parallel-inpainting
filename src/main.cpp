#include <cstddef>
#include <iostream>

#include "../include/inpaint.h"
#include "../include/CycleTimer.h"

using namespace std;

int main()
{
    // read image
    cv::Mat prunedImg = cv::imread("../images/forest_pruned.bmp", cv::IMREAD_COLOR);

    printf("Image dimensions :\n");
    printf("Height: %d, Width: %d\n", prunedImg.size().height, prunedImg.size().width);
    
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

    // auto metric = PatchSSDDistanceMetric(3);
    // double startTime = CycleTimer::currentSeconds();
    // auto cpu_output = Inpainting(prunedImg, mask, &metric, false).run(false, false);
    // double endTime = CycleTimer::currentSeconds();

    // printf("CPU Processing time: %lfs\n", endTime - startTime);

    startTime = CycleTimer::currentSeconds();
    auto gpu_output = Inpainting(prunedImg, mask, &metric, true).run(false, false);
    endTime = CycleTimer::currentSeconds();

    printf("GPU Processing time: %lfs\n", endTime - startTime);

    cv::Mat diff;
    cv::absdiff(cpu_output, gpu_output, diff);
    double max_diff;
    cv::minMaxLoc(diff.reshape(1), nullptr, &max_diff); // not tracking min diff
    cv::Scalar mean, stddev;
    cv::meanStdDev(diff, mean, stddev);
    printf("Max diff: %lf, Mean diff: %lf, Stddev: %lf\n", max_diff, mean[0], stddev[0]);

    // cv::imshow("Result", result);
    // cv::waitKey();

    // std::string ctx = is_gpu_enabled ? "GPU" : "CPU";
    // std::string output_file = "output_" + ctx + ".png";
    // bool success = cv::imwrite("../images/" + output_file, result);

    return 0;
}