#pragma once

#include "cuda_buffers.h"

extern "C" {
void launch_nnf_minimize(CudaNNFDeviceBuffers *bufs, int4 *d_field_ptr,
						 int4 *d_field_scratch, int *h_field_ptr,
						 const HostImageBuffers &src,
						 const HostImageBuffers &tgt, bool has_gmask,
						 int patch_size, int nr_pass, unsigned int random_seed);

void launch_nnf_randomize(CudaNNFDeviceBuffers *bufs, int4 *d_field_ptr,
						  const HostImageBuffers &src,
						  const HostImageBuffers &tgt, bool has_gmask,
						  int patch_size, int max_retry, bool reset,
						  unsigned int seed, cudaStream_t stream);

void launch_nnf_initialize_from(CudaNNFDeviceBuffers *bufs, int4 *d_field_ptr,
								const int4 *other_d_field_ptr,
								const HostImageBuffers &src,
								const HostImageBuffers &tgt, int other_src_h,
								int other_src_w, bool has_gmask, int patch_size,
								int max_retry, unsigned int seed, cudaStream_t stream);

void launch_nnf_set_identity(CudaNNFDeviceBuffers *bufs, int4 *d_field_ptr,
							 const HostImageBuffers &src, bool has_gmask,
							 int patch_size);

void launch_em_iteration(CudaNNFDeviceBuffers *bufs, int src_h, int src_w,
						 int tgt_h, int tgt_w, bool has_gmask, int patch_size,
						 cudaStream_t stream);
} // extern "C"

void download_target_pixels(CudaNNFDeviceBuffers *bufs, uchar4 *host_dst,
							int tgt_pixels, cudaStream_t stream);