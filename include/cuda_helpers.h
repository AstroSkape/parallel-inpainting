#pragma once
// #define DEBUG

void cuda_device_sync();

#ifdef DEBUG
#define LOG(fmt, ...) printf("[DEBUG] " fmt, ##__VA_ARGS__)
#else
#define LOG(fmt, ...)
#endif