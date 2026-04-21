#pragma once

#include "cuda_buffers.h"

extern "C" {
void launch_nnf_minimize(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
						 int *d_field_scratch, int *h_field_ptr,
						 const HostImageBuffers &src,
						 const HostImageBuffers &tgt, bool has_gmask,
						 int patch_size, int nr_pass, unsigned int random_seed);

void launch_nnf_randomize(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
						  const HostImageBuffers &src,
						  const HostImageBuffers &tgt, bool has_gmask,
						  int patch_size, int max_retry, bool reset,
						  unsigned int seed);

void launch_nnf_initialize_from(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
								const int *other_d_field_ptr,
								const HostImageBuffers &src,
								const HostImageBuffers &tgt, int other_src_h,
								int other_src_w, bool has_gmask, int patch_size,
								int max_retry, unsigned int seed);

void launch_nnf_set_identity(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
							 const HostImageBuffers &src, bool has_gmask,
							 int patch_size);
} // extern "C"