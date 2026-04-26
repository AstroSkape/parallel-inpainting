#pragma once

#include "cuda_buffers.h"
#include <vector>

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

void upload_similarity_table(const std::vector<double> &table);

void launch_expectation_step(double *d_vote, const int *d_field_ptr, 
    const PixelData *d_src, const PixelData *d_nnf_src, const PixelData *d_nnf_tgt,
    bool has_gmask, int src_h, int src_w, int nnf_src_h, int nnf_src_w,
    int nnf_tgt_h, int nnf_tgt_w, bool source2target, bool upscaled,
    int patch_size);

void launch_maximization_step(unsigned char *d_target_img,
    unsigned char *d_target_mask, const unsigned char *d_target_gmask,
    const double *d_vote, bool has_gmask, int tgt_h, int tgt_w);