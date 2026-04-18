#include <cuda_runtime.h>
#include "../include/cuda_helpers.cuh"

void cuda_device_sync()
{
    cudaCheckError(cudaDeviceSynchronize());
}