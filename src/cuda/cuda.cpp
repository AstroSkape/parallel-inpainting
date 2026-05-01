#include "cuda/cuda_buffers.h"
#include "cuda/cuda_helpers.cuh"
#include "cuda/nnf_cuda.h"
#include "patchmatch/inpaint.h"
#include "patchmatch/nnf.h"

HostImageBuffers
NearestNeighborField::_make_host_buffers(const MaskedImage &img,
										 bool include_pixels) const {
	HostImageBuffers buf;
	auto size = img.size();
	buf.height = size.height;
	buf.width = size.width;

	img.compute_image_gradients();
	const int pixels = size.height * size.width;
	const bool has_gmask = !img.global_mask().empty();

	const unsigned char *img_ptr = img.image().ptr<unsigned char>(0, 0);
	const unsigned char *gx_ptr = img.gradx().ptr<unsigned char>(0, 0);
	const unsigned char *gy_ptr = img.grady().ptr<unsigned char>(0, 0);
	const unsigned char *mask_ptr = img.mask().ptr<unsigned char>(0, 0);
	const unsigned char *gmask_ptr =
		has_gmask ? img.global_mask().ptr<unsigned char>(0, 0) : nullptr;

	buf.pack_pixel_data_from(img_ptr, gx_ptr, gy_ptr, mask_ptr, gmask_ptr,
							 pixels);

	return buf;
}

void NearestNeighborField::minimize_cuda(int nr_pass,
										 CudaNNFDeviceBuffers *bufs,
										 int4 *d_field_ptr,
										 int4 *d_field_scratch) {
	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	bool include_pixels = true;
	HostImageBuffers src = _make_host_buffers(m_source, include_pixels);
	HostImageBuffers tgt = _make_host_buffers(m_target, include_pixels);

	launch_nnf_minimize(bufs, d_field_ptr, d_field_scratch,
						m_field.ptr<int>(0, 0), src, tgt, has_gmask,
						m_distance_metric->patch_size(), nr_pass,
						(unsigned int)rand());
}

void NearestNeighborField::initialize_cuda_randomize(CudaNNFDeviceBuffers *bufs,
													 int4 *d_field_ptr,
													 int max_retry,
													 unsigned int seed,
													 cudaStream_t stream) {
	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	bool include_pixels = true;
	HostImageBuffers src = _make_host_buffers(m_source, include_pixels);
	HostImageBuffers tgt = _make_host_buffers(m_target, include_pixels);

	launch_nnf_randomize(bufs, d_field_ptr, src, tgt, has_gmask,
						 m_distance_metric->patch_size(), max_retry, true, seed,
						 stream);
}

void NearestNeighborField::initialize_cuda_from(
	CudaNNFDeviceBuffers *bufs, int4 *d_field_ptr, const int4 *other_d_field_ptr,
	cv::Size other_source_size, int max_retry, unsigned int seed,
	cudaStream_t stream) {
	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	bool include_pixels = true;
	HostImageBuffers src = _make_host_buffers(m_source, include_pixels);
	HostImageBuffers tgt = _make_host_buffers(m_target, include_pixels);

	launch_nnf_initialize_from(
		bufs, d_field_ptr, other_d_field_ptr, src, tgt,
		other_source_size.height, other_source_size.width, has_gmask,
		m_distance_metric->patch_size(), max_retry, seed, stream);
}

void NearestNeighborField::set_identity_cuda(CudaNNFDeviceBuffers *bufs,
											 int4 *d_field_ptr,
											 const MaskedImage &mask_source) {
	bool has_gmask = !m_source.global_mask().empty();

	HostImageBuffers src = _make_host_buffers(mask_source, false);

	launch_nnf_set_identity(bufs, d_field_ptr, src, has_gmask,
							m_distance_metric->patch_size());
}

// copies the pixel info from device back to host
void download_target_pixels(CudaNNFDeviceBuffers *bufs, uchar4 *host_dst,
							int tgt_pixels, cudaStream_t stream) {
	cudaCheckError(cudaMemcpyAsync(host_dst, bufs->tgt_bufs.rgb_mask,
								   tgt_pixels * sizeof(uchar4),
								   cudaMemcpyDeviceToHost, stream));
}