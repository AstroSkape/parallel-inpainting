#include <cstddef>
#include <iostream>

#include "../include/inpaint.h"
#include "../include/CycleTimer.h"

using namespace std;

cv::Mat createMask(cv::Mat imageWithHole)
{
    // create mask
    cv::Mat mask = cv::Mat(imageWithHole.size(), CV_8UC1);
    for (size_t i = 0; i < imageWithHole.size().height; i++)
    {
        for (size_t j = 0; j < imageWithHole.size().width; j++)
        {
            cv::Vec3b pixel = imageWithHole.at<cv::Vec3b>(i, j);
            if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
                mask.at<unsigned char>(i, j) = 1;
        }
    }
    return mask;
}

double getExecutionTime(bool is_gpu_enabled, cv::Mat &imageWithHole, cv::Mat &mask, PatchDistanceMetric &metric)
{
    double startTime = CycleTimer::currentSeconds();
    auto output = Inpainting(imageWithHole, mask, &metric, is_gpu_enabled).run(false, false);
    double endTime = CycleTimer::currentSeconds();

    std::string ctx = is_gpu_enabled ? "GPU" : "CPU";
    std::string output_file = "output_" + ctx + ".png";
    bool success = cv::imwrite("../images/" + output_file, output);

    return endTime - startTime;
}

int main()
{
    // read image
    cv::Mat prunedImg = cv::imread("../images/forest_pruned.bmp", cv::IMREAD_COLOR);
    auto mask = createMask(prunedImg);
    auto metric = PatchSSDDistanceMetric(3);

    auto serial = getExecutionTime(false, prunedImg, mask, metric);
    auto parallel = getExecutionTime(true, prunedImg, mask, metric);
    double speedup = serial / parallel;
    
    printf("Image dimensions :\n");
    printf("Height: %d, Width: %d\n", prunedImg.size().height, prunedImg.size().width);
    printf("CPU Processing time: %lfs\n", serial);
    printf("GPU Processing time: %lfs\n", parallel);
    printf("Speedup observed: %lf\n", speedup);

    return 0;
}