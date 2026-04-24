#include <cuda_runtime.h>
#include "cuda/cuda_helpers.cuh"
#include "cuda/cuda_buffers.h"

void cuda_device_sync()
{
    cudaCheckError(cudaDeviceSynchronize());
}

void HostImageBuffers::pack_pixel_data_from(const unsigned char *img,
											const unsigned char *gx_in,
											const unsigned char *gy_in,
											const unsigned char *mask,
											const unsigned char *gmask,
											int pixels) {
	if (!rgb_mask)
		rgb_mask = new uchar4[pixels];
	if (!gx)
		gx = new uchar4[pixels];
	if (!gy)
		gy = new uchar4[pixels];

	const bool has_gmask = (gmask != nullptr);

	#pragma omp parallel for schedule(static) num_threads(8)  
	for (int i = 0; i < pixels; ++i) {
        uchar4 rm;
        rm.x = img[i * 3 + 0];
        rm.y = img[i * 3 + 1];
        rm.z = img[i * 3 + 2];
        // Pack both mask bits into .w: bit 0 = mask, bit 1 = gmask
        unsigned char packed_mask = mask[i] ? 1u : 0u;
        if (has_gmask && gmask[i]) packed_mask |= 2u;
        rm.w = packed_mask;
        rgb_mask[i] = rm;

        uchar4 gxv;
        gxv.x = gx_in[i * 3 + 0];
        gxv.y = gx_in[i * 3 + 1];
        gxv.z = gx_in[i * 3 + 2];
        gxv.w = 0;
        gx[i] = gxv;

        uchar4 gyv;
        gyv.x = gy_in[i * 3 + 0];
        gyv.y = gy_in[i * 3 + 1];
        gyv.z = gy_in[i * 3 + 2];
        gyv.w = 0;
        gy[i] = gyv;
    }
}

void HostImageBuffers::free_pixel_data() {
    delete[] rgb_mask; rgb_mask = nullptr;
    delete[] gx;       gx       = nullptr;
    delete[] gy;       gy       = nullptr;
}