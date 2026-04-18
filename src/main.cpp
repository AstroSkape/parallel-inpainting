#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include "../include/inpaint.h"
#include "masked_image.h"
#include <iostream>
#include <string>

using namespace std;

void check_correctness(const std::string& serial_path, 
                        const std::string& parallel_path) {
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
    int total_pixels = serial.rows * serial.cols;
    int differing_pixels = 0;
    double total_diff = 0;
    int max_diff = 0;
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

    double avg_diff = total_diff / total_pixels;

    double variance = 0;
    for (double d : diffs) {
        variance += (d - avg_diff) * (d - avg_diff);
    }
    double stddev = std::sqrt(variance / total_pixels);

    // PSNR computation
    cv::Mat diff_mat;
    cv::absdiff(serial, parallel, diff_mat);
    diff_mat.convertTo(diff_mat, CV_32F);
    double mse = cv::mean(diff_mat.mul(diff_mat))[0];
    double psnr = (mse > 0) ? 10.0 * log10(255.0 * 255.0 / mse) : 100.0;

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

int main(int argc, char* argv[]) {
    // default to 1 thread (serial) if not specified
    int num_threads = 1;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    
    omp_set_num_threads(num_threads);
    std::cout << "Running with " << num_threads << " thread(s)" << std::endl;

    // determine output filename based on thread count
    std::string output_filename;
    if (num_threads == 1) {
        output_filename = "output-serial.png";
    } else {
        output_filename = "output-parallel-" + std::to_string(num_threads) + ".png";
    }

    // read image and create mask
    cv::Mat prunedImg = cv::imread("../images/forest_pruned.bmp", cv::IMREAD_COLOR);
    cv::Mat mask = cv::Mat(prunedImg.size(), CV_8UC1);
    for (size_t i = 0; i < prunedImg.size().height; i++) {
        for (size_t j = 0; j < prunedImg.size().width; j++) {
            cv::Vec3b pixel = prunedImg.at<cv::Vec3b>(i, j);
            if (pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255)
                mask.at<unsigned char>(i, j) = 1;
        }
    }
    auto wall_start = std::chrono::high_resolution_clock::now();

    auto metric = PatchSSDDistanceMetric(3);
    auto result = Inpainting(prunedImg, mask, &metric).run(true, false);
    auto wall_end = std::chrono::high_resolution_clock::now();

    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    std::cout << "Total processing time: " << wall_ms / 1000.0 << "s" << std::endl;

    cv::imwrite(output_filename, result);
    std::cout << "Output written to: " << output_filename << std::endl;

    check_correctness("output-serial.png", output_filename);
    return 0;
}