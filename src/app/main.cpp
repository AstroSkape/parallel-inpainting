#include <cstddef>
#include <iostream>

#include "patchmatch/inpaint.h"
#include "utils/cycle_timer.h"

using namespace std;

double startTime;

void check_correctness(const std::string& serial_path, 
                        const std::string& parallel_path,
                        int& total_pixels,
                    int &differing_pixels, double &total_diff,
                int &max_diff, double &avg_diff, double &stddev, double &psnr) {
    cv::Mat serial = cv::imread(serial_path, cv::IMREAD_COLOR);
    cv::Mat parallel = cv::imread(parallel_path, cv::IMREAD_COLOR);

    if (serial.empty() || parallel.empty()) {
        std::cout << "Could not load images for comparison" << std::endl;
        return;
    }

    if (serial.size() != parallel.size()) {
        std::cout << "Image sizes differ!" << std::endl;
        return;
    }

    // pixel difference stats
    total_pixels = serial.rows * serial.cols;
    differing_pixels = 0;
    total_diff = 0;
    max_diff = 0;
    std::vector<double> diffs;
    diffs.reserve(total_pixels);

    for (int i = 0; i < serial.rows; i++) {
        for (int j = 0; j < serial.cols; j++) {
            cv::Vec3b s = serial.at<cv::Vec3b>(i, j);
            cv::Vec3b p = parallel.at<cv::Vec3b>(i, j);
            
            int diff = 0;
            for (int c = 0; c < 3; c++) {
                diff += abs((int)s[c] - (int)p[c]);
            }
            
            if (diff > 0) differing_pixels++;
            total_diff += diff;
            max_diff = std::max(max_diff, diff);
            diffs.push_back(diff);
        }
    }

    avg_diff = total_diff / total_pixels;

    double variance = 0;
    for (double d : diffs) {
        variance += (d - avg_diff) * (d - avg_diff);
    }
    stddev = std::sqrt(variance / total_pixels);

    // PSNR computation
    cv::Mat diff_mat;
    cv::absdiff(serial, parallel, diff_mat);
    diff_mat.convertTo(diff_mat, CV_32F);
    double mse = cv::mean(diff_mat.mul(diff_mat))[0];
    psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 100.0;

    std::cout << "=== Correctness Check ===" << std::endl;
    std::cout << "Total pixels:      " << total_pixels << std::endl;
    std::cout << "Differing pixels:  " << differing_pixels 
              << " (" << (100.0 * differing_pixels / total_pixels) << "%)" << std::endl;
    std::cout << "Average diff:      " << avg_diff << std::endl;
    std::cout << "Max diff:          " << max_diff << std::endl;
    std::cout << "Stddev diff:       " << stddev << std::endl;
    std::cout << "PSNR (serial vs parallel): " << psnr << " dB" << std::endl;
    
    if (psnr > 35.0) {
        std::cout << "Quality: GOOD — outputs are visually equivalent" << std::endl;
    } else if (psnr > 25.0) {
        std::cout << "Quality: ACCEPTABLE — minor differences" << std::endl;
    } else {
        std::cout << "Quality: POOR — significant differences, check implementation" << std::endl;
    }
}

cv::Mat createMask(cv::Mat imageWithHole)
{
    // create mask
    cv::Mat mask = cv::Mat::zeros(imageWithHole.size(), CV_8UC1);
    for (size_t i = 0; i < imageWithHole.size().height; i++)
    {
        for (size_t j = 0; j < imageWithHole.size().width; j++)
        {
            cv::Vec3b pixel = imageWithHole.at<cv::Vec3b>(i, j);
            // fixes mask edges that are almost white 
            if (pixel[0] >= 240 && pixel[1] >= 240 && pixel[2] >= 240)
                mask.at<unsigned char>(i, j) = 1;
        }
    }
    // extends the mask slightly by a couple pixels
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1,-1), 2);
    cv::imwrite("../images/mask_debug.png", mask * 255);
    return mask;
}

double getExecutionTime(bool is_gpu_enabled, cv::Mat &imageWithHole, std::string input_path, cv::Mat &mask, PatchDistanceMetric &metric)
{
    startTime = CycleTimer::currentSeconds();
    auto inpainter = Inpainting(imageWithHole, mask, &metric, is_gpu_enabled);
    double initEndTime = CycleTimer::currentSeconds();
    auto output = inpainter.run(false, false);
    double endTime = CycleTimer::currentSeconds();

    std::string ctx = is_gpu_enabled ? "GPU" : "CPU";
    std::string output_file = input_path + "_output_" + ctx + ".png";
    bool success = cv::imwrite(output_file, output);

    return endTime - initEndTime;
}

int main(int argc, char* argv[])
{
     if (argc < 2) {
        std::cerr << "Usage: ./inpaint <input_path>\n";
        return 1;
    }
    std::string input_path  = argv[1];

    cv::Mat prunedImg = cv::imread(input_path);
    // read image
    // cv::Mat prunedImg = cv::imread("../images/forest_pruned.bmp", cv::IMREAD_COLOR);
    auto mask = createMask(prunedImg);
    auto metric = PatchSSDDistanceMetric(3);

    auto serial = getExecutionTime(false, prunedImg, input_path, mask, metric);
    auto parallel = getExecutionTime(true, prunedImg, input_path, mask, metric);
    int total_pixels = 0;
    int differing_pixels = 0;
    int max_diff = 0;
    double total_diff = 0;
    double avg_diff = 0;
    double stddev = 0;
    double psnr = 0;
    check_correctness(input_path + "_output_CPU.png", input_path + "_output_GPU.png",
    total_pixels, differing_pixels, total_diff, max_diff, avg_diff, stddev, psnr);
    double speedup = serial / parallel;
    
    // print JSON-parseable metrics to stdout
    printf("Image dimensions :\n");
    printf("Height: %d, Width: %d\n", prunedImg.size().height, prunedImg.size().width);
    printf("CPU Processing time: %lfs\n", serial);
    printf("GPU Processing time: %lfs\n", parallel);
    printf("Speedup observed: %lf\n", speedup);


    printf("METRICS_JSON:{\"total_pixels\":%d,\"differing_pixels\":%d,"
        "\"differing_pct\":%.4f,\"avg_diff\":%.4f,\"max_diff\":%d,"
        "\"stddev\":%.4f,\"psnr\":%.4f,"
        "\"cpu_time\":%.6f,\"gpu_time\":%.6f,\"speedup\":%.6f}\n",
        total_pixels, differing_pixels,
        100.0 * differing_pixels / total_pixels,
        avg_diff, max_diff, stddev, psnr,
        serial, parallel, speedup);
    return 0;
}