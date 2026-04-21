#include <cuda_runtime.h>
#include "cuda/cuda_helpers.cuh"

void cuda_device_sync()
{
    cudaCheckError(cudaDeviceSynchronize());
}