#include "cuda_helpers.h"
#include "inpaint.h"
#include "nnf.h"

extern "C" void launch_nnf_minimize(
	CudaNNFDeviceBuffers *bufs, int *d_field_ptr, int *d_field_scratch,
	int *h_field_ptr, const HostImageBuffers &src, const HostImageBuffers &tgt,
	bool has_gmask, int patch_size, int nr_pass, unsigned int random_seed);

extern "C" void
launch_nnf_randomize(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
					 const HostImageBuffers &src, const HostImageBuffers &tgt,
					 bool has_gmask, int patch_size, int max_retry, bool reset,
					 unsigned int seed);

extern "C" void launch_nnf_initialize_from(
	CudaNNFDeviceBuffers *bufs, int *d_field_ptr, const int *other_d_field_ptr,
	const HostImageBuffers &src, const HostImageBuffers &tgt, int other_src_h,
	int other_src_w, bool has_gmask, int patch_size, int max_retry,
	unsigned int seed);

extern "C" void launch_nnf_set_identity(CudaNNFDeviceBuffers *bufs,
										int *d_field_ptr,
										const HostImageBuffers &src,
										bool has_gmask, int patch_size);

void NearestNeighborField::minimize_cuda(int nr_pass,
										 CudaNNFDeviceBuffers *bufs,
										 int *d_field_ptr,
										 int *d_field_scratch) {
	m_source.compute_image_gradients();
	m_target.compute_image_gradients();

	cv::Size src_size = source_size();
	cv::Size tgt_size = target_size();

	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	HostImageBuffers src{
		m_source.image().ptr<unsigned char>(0, 0),
		m_source.gradx().ptr<unsigned char>(0, 0),
		m_source.grady().ptr<unsigned char>(0, 0),
		m_source.mask().ptr<unsigned char>(0, 0),
		has_gmask ? m_source.global_mask().ptr<unsigned char>(0, 0) : nullptr,
		src_size.height,
		src_size.width,
	};

	HostImageBuffers tgt{
		m_target.image().ptr<unsigned char>(0, 0),
		m_target.gradx().ptr<unsigned char>(0, 0),
		m_target.grady().ptr<unsigned char>(0, 0),
		m_target.mask().ptr<unsigned char>(0, 0),
		has_gmask ? m_target.global_mask().ptr<unsigned char>(0, 0) : nullptr,
		tgt_size.height,
		tgt_size.width,
	};

	launch_nnf_minimize(bufs, d_field_ptr, d_field_scratch,
						m_field.ptr<int>(0, 0), src, tgt, has_gmask,
						m_distance_metric->patch_size(), nr_pass,
						(unsigned int)rand());
}

void NearestNeighborField::initialize_cuda_randomize(CudaNNFDeviceBuffers *bufs,
													 int *d_field_ptr,
													 int max_retry,
													 unsigned int seed) {
	m_source.compute_image_gradients();
	m_target.compute_image_gradients();

	cv::Size src_size = source_size();
	cv::Size tgt_size = target_size();

	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	HostImageBuffers src{
		m_source.image().ptr<unsigned char>(0, 0),
		m_source.gradx().ptr<unsigned char>(0, 0),
		m_source.grady().ptr<unsigned char>(0, 0),
		m_source.mask().ptr<unsigned char>(0, 0),
		has_gmask ? m_source.global_mask().ptr<unsigned char>(0, 0) : nullptr,
		src_size.height,
		src_size.width,
	};

	HostImageBuffers tgt{
		m_target.image().ptr<unsigned char>(0, 0),
		m_target.gradx().ptr<unsigned char>(0, 0),
		m_target.grady().ptr<unsigned char>(0, 0),
		m_target.mask().ptr<unsigned char>(0, 0),
		has_gmask ? m_target.global_mask().ptr<unsigned char>(0, 0) : nullptr,
		tgt_size.height,
		tgt_size.width,
	};

	launch_nnf_randomize(bufs, d_field_ptr, src, tgt, has_gmask,
						 m_distance_metric->patch_size(), max_retry, true,
						 seed);

	LOG("randomize kernel returned\n");
}

void NearestNeighborField::initialize_cuda_from(
	CudaNNFDeviceBuffers *bufs, int *d_field_ptr, const int *other_d_field_ptr,
	cv::Size other_source_size, int max_retry, unsigned int seed) {

	m_source.compute_image_gradients();
	m_target.compute_image_gradients();

	cv::Size src_size = source_size();
	cv::Size tgt_size = target_size();

	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	HostImageBuffers src{
		m_source.image().ptr<unsigned char>(0, 0),
		m_source.gradx().ptr<unsigned char>(0, 0),
		m_source.grady().ptr<unsigned char>(0, 0),
		m_source.mask().ptr<unsigned char>(0, 0),
		has_gmask ? m_source.global_mask().ptr<unsigned char>(0, 0) : nullptr,
		src_size.height,
		src_size.width,
	};

	HostImageBuffers tgt{
		m_target.image().ptr<unsigned char>(0, 0),
		m_target.gradx().ptr<unsigned char>(0, 0),
		m_target.grady().ptr<unsigned char>(0, 0),
		m_target.mask().ptr<unsigned char>(0, 0),
		has_gmask ? m_target.global_mask().ptr<unsigned char>(0, 0) : nullptr,
		tgt_size.height,
		tgt_size.width,
	};

	launch_nnf_initialize_from(
		bufs, d_field_ptr, other_d_field_ptr, src, tgt,
		other_source_size.height, other_source_size.width, has_gmask,
		m_distance_metric->patch_size(), max_retry, seed);
}

void NearestNeighborField::set_identity_cuda(CudaNNFDeviceBuffers *bufs,
											 int *d_field_ptr,
											 const MaskedImage &mask_source) {
	cv::Size s = source_size();
	bool has_gmask = !m_source.global_mask().empty();

	HostImageBuffers src{
		nullptr, // unused
		nullptr, // unused
		nullptr, // unused
		mask_source.mask().ptr<unsigned char>(0, 0),
		has_gmask ? mask_source.global_mask().ptr<unsigned char>(0, 0)
				  : nullptr,
		s.height,
		s.width,
	};

	launch_nnf_set_identity(bufs, d_field_ptr, src, has_gmask,
							m_distance_metric->patch_size());
}