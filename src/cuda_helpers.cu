#include <cuda_runtime.h>

void cuda_device_sync()
{
    cudaDeviceSynchronize();
}