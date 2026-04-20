#include "../include/cuda_helpers.h"
#include "../include/inpaint.h"
#include "../include/nnf.h"

extern "C" void launch_nnf_minimize(CudaNNFDeviceBuffers *bufs,
									int *device_field_ptr, int *field_ptr,
									const HostImageBuffers &src,
									const HostImageBuffers &tgt, bool has_gmask,
									int patch_size, int nr_pass,
									unsigned int random_seed);

void NearestNeighborField::minimize_cuda(int nr_pass,
										 CudaNNFDeviceBuffers *bufs,
										 int *d_field_ptr) {
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

	launch_nnf_minimize(bufs, d_field_ptr, m_field.ptr<int>(0, 0), src, tgt, has_gmask,
						m_distance_metric->patch_size(), nr_pass,
						(unsigned int)rand());
}