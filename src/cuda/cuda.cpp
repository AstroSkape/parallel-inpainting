#include "cuda/cuda_buffers.h"
#include "patchmatch/inpaint.h"
#include "patchmatch/nnf.h"
#include "cuda/nnf_cuda.h"


HostImageBuffers NearestNeighborField::_make_host_buffers(const MaskedImage &img,
										bool include_pixels) const {
    cv::Size size = img.size();

	bool has_gmask =
		!img.global_mask().empty() && !img.global_mask().empty();

	HostImageBuffers buf;
	buf.height = size.height;
	buf.width = size.width;
	buf.mask = img.mask().ptr<unsigned char>(0, 0);
	buf.gmask = has_gmask ? img.global_mask().ptr<unsigned char>(0, 0) : nullptr;

	if (include_pixels) {
		img.compute_image_gradients();
		buf.img = img.image().ptr<unsigned char>(0, 0);
		buf.gx = img.gradx().ptr<unsigned char>(0, 0);
		buf.gy = img.grady().ptr<unsigned char>(0, 0);
	}
	return buf;
}


void NearestNeighborField::minimize_cuda(int nr_pass,
										 CudaNNFDeviceBuffers *bufs,
										 int *d_field_ptr,
										 int *d_field_scratch) {
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
													 int *d_field_ptr,
													 int max_retry,
													 unsigned int seed) {
	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	bool include_pixels = true;
	HostImageBuffers src = _make_host_buffers(m_source, include_pixels);
	HostImageBuffers tgt = _make_host_buffers(m_target, include_pixels);

	launch_nnf_randomize(bufs, d_field_ptr, src, tgt, has_gmask,
						 m_distance_metric->patch_size(), max_retry, true,
						 seed);

	LOG("randomize kernel returned\n");
}

void NearestNeighborField::initialize_cuda_from(
	CudaNNFDeviceBuffers *bufs, int *d_field_ptr, const int *other_d_field_ptr,
	cv::Size other_source_size, int max_retry, unsigned int seed) {
	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	bool include_pixels = true;
	HostImageBuffers src = _make_host_buffers(m_source, include_pixels);
	HostImageBuffers tgt = _make_host_buffers(m_target, include_pixels);

	launch_nnf_initialize_from(
		bufs, d_field_ptr, other_d_field_ptr, src, tgt,
		other_source_size.height, other_source_size.width, has_gmask,
		m_distance_metric->patch_size(), max_retry, seed);
}

void NearestNeighborField::set_identity_cuda(CudaNNFDeviceBuffers *bufs,
											 int *d_field_ptr,
											 const MaskedImage &mask_source) {
	bool has_gmask = !m_source.global_mask().empty();

	HostImageBuffers src =
		_make_host_buffers(mask_source, false);

	launch_nnf_set_identity(bufs, d_field_ptr, src, has_gmask,
							m_distance_metric->patch_size());
}