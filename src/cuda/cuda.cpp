#include "cuda/cuda_buffers.h"
#include "cuda/nnf_cuda.h"
#include "patchmatch/inpaint.h"
#include "patchmatch/nnf.h"

std::vector<PixelData>
NearestNeighborField::_pack_pixel_data(const MaskedImage &img,
									   bool include_pixels) const {

	auto size = img.size();
	int num_pixels = size.height * size.width;
	bool has_gmask = !img.global_mask().empty();

	std::vector<PixelData> buf(num_pixels);

	const unsigned char *mask_ptr = img.mask().ptr<unsigned char>(0, 0);
	const unsigned char *gmask_ptr =
		has_gmask ? img.global_mask().ptr<unsigned char>(0, 0) : nullptr;

	const unsigned char *img_ptr = nullptr;
	const unsigned char *gx_ptr = nullptr;
	const unsigned char *gy_ptr = nullptr;

	if (include_pixels) {
		img.compute_image_gradients();
		img_ptr = img.image().ptr<unsigned char>(0, 0);
		gx_ptr = img.gradx().ptr<unsigned char>(0, 0);
		gy_ptr = img.grady().ptr<unsigned char>(0, 0);
	}

	for (int i = 0; i < num_pixels; i++) {
		PixelData &p = buf[i];
		memset(&p, 0, sizeof(PixelData));

		p.mask = mask_ptr[i];
		p.gmask = gmask_ptr ? gmask_ptr[i] : 0;

		if (include_pixels) {
			p.rgb = make_uchar3(img_ptr[i * 3], img_ptr[i * 3 + 1],
								img_ptr[i * 3 + 2]);
			p.gx = make_uchar3(gx_ptr[i * 3], gx_ptr[i * 3 + 1],
							   gx_ptr[i * 3 + 2]);
			p.gy = make_uchar3(gy_ptr[i * 3], gy_ptr[i * 3 + 1],
							   gy_ptr[i * 3 + 2]);
		}
	}

	return buf;
}

HostImageBuffers
NearestNeighborField::_make_host_buffers(std::vector<PixelData> &packed,
										 const MaskedImage &img) const {
	HostImageBuffers buf;
	buf.data = packed.data();
	buf.height = img.size().height;
	buf.width = img.size().width;
	return buf;
}

void NearestNeighborField::minimize_cuda(int nr_pass,
										 CudaNNFDeviceBuffers *bufs,
										 int *d_field_ptr,
										 int *d_field_scratch) {
	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	bool include_pixels = true;
	auto src_packed = _pack_pixel_data(m_source, include_pixels);
	auto tgt_packed = _pack_pixel_data(m_target, include_pixels);

	HostImageBuffers src = _make_host_buffers(src_packed, m_source);
	HostImageBuffers tgt = _make_host_buffers(tgt_packed, m_target);

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
	auto src_packed = _pack_pixel_data(m_source, include_pixels);
	auto tgt_packed = _pack_pixel_data(m_target, include_pixels);

	HostImageBuffers src = _make_host_buffers(src_packed, m_source);
	HostImageBuffers tgt = _make_host_buffers(tgt_packed, m_target);

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
	auto src_packed = _pack_pixel_data(m_source, include_pixels);
	auto tgt_packed = _pack_pixel_data(m_target, include_pixels);

	HostImageBuffers src = _make_host_buffers(src_packed, m_source);
	HostImageBuffers tgt = _make_host_buffers(tgt_packed, m_target);

	launch_nnf_initialize_from(
		bufs, d_field_ptr, other_d_field_ptr, src, tgt,
		other_source_size.height, other_source_size.width, has_gmask,
		m_distance_metric->patch_size(), max_retry, seed);
}

void NearestNeighborField::set_identity_cuda(CudaNNFDeviceBuffers *bufs,
											 int *d_field_ptr,
											 const MaskedImage &mask_source) {
	bool has_gmask = !m_source.global_mask().empty();

	auto src_packed = _pack_pixel_data(mask_source, false);
	HostImageBuffers src = _make_host_buffers(src_packed, mask_source);

	launch_nnf_set_identity(bufs, d_field_ptr, src, has_gmask,
							m_distance_metric->patch_size());
}