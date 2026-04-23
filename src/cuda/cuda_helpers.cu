#include <cuda_runtime.h>
#include "cuda/cuda_helpers.cuh"
#include "cuda/cuda_buffers.h"

void cuda_device_sync()
{
    cudaCheckError(cudaDeviceSynchronize());
}

void HostImageBuffers::pack_pixel_data_from(
    const unsigned char *img, const unsigned char *gx, const unsigned char *gy,
    const unsigned char *mask, const unsigned char *gmask, int pixels)
{
    if (!data) data = new PixelData[pixels];  // or use cudaMallocHost for pinned
    for (int i = 0; i < pixels; i++) {
        data[i].rgb   = {img[i*3+0], img[i*3+1], img[i*3+2]};
        data[i].gx    = {gx[i*3+0],  gx[i*3+1],  gx[i*3+2]};
        data[i].gy    = {gy[i*3+0],  gy[i*3+1],  gy[i*3+2]};
        data[i].mask  = mask[i];
        data[i].gmask = gmask ? gmask[i] : 0;
    }
}