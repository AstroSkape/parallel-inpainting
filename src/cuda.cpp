#include "../include/nnf.h"

extern "C" void launch_nnf_minimize(
	CudaNNFDeviceBuffers *bufs, int *field_ptr, const unsigned char *src_img,
	const unsigned char *tgt_img, const unsigned char *src_gx,
	const unsigned char *src_gy, const unsigned char *tgt_gx,
	const unsigned char *tgt_gy, const unsigned char *src_mask,
	const unsigned char *tgt_mask, const unsigned char *src_gmask,
	const unsigned char *tgt_gmask, bool has_gmask, int src_height,
	int src_width, int tgt_height, int tgt_width, int patch_size, int nr_pass,
	unsigned int random_seed);

void NearestNeighborField::minimize_cuda(int nr_pass, CudaNNFDeviceBuffers *bufs) {
	m_source.compute_image_gradients();
	m_target.compute_image_gradients();

	cv::Size src_size = source_size();
	cv::Size tgt_size = target_size();

	bool has_gmask =
		!m_source.global_mask().empty() && !m_target.global_mask().empty();

	launch_nnf_minimize(
		bufs, m_field.ptr<int>(0, 0), m_source.image().ptr<unsigned char>(0, 0),
		m_target.image().ptr<unsigned char>(0, 0),
		m_source.gradx().ptr<unsigned char>(0, 0),
		m_source.grady().ptr<unsigned char>(0, 0),
		m_target.gradx().ptr<unsigned char>(0, 0),
		m_target.grady().ptr<unsigned char>(0, 0),
		m_source.mask().ptr<unsigned char>(0, 0),
		m_target.mask().ptr<unsigned char>(0, 0),
		has_gmask ? m_source.global_mask().ptr<unsigned char>(0, 0) : nullptr,
		has_gmask ? m_target.global_mask().ptr<unsigned char>(0, 0) : nullptr,
		has_gmask, src_size.height, src_size.width, tgt_size.height,
		tgt_size.width, m_distance_metric->patch_size(), nr_pass,
		(unsigned int)rand());
}