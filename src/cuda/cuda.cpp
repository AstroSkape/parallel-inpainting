#include "cuda/cuda_buffers.h"
#include "cuda/nnf_cuda.h"
#include "cuda/cuda_helpers.cuh"
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

void Inpainting::_expectation_step_cuda(const NearestNeighborField &nnf,
                                        bool source2target, const MaskedImage &source,
                                        bool upscaled, CudaNNFDeviceBuffers *cuda_bufs, cv::Mat &vote) {

    bool has_gmask = !nnf.source().global_mask().empty();
    bool include_pixels = true;
	int *d_field_ptr = source2target ? m_cuda_buffers.s2t_curr : m_cuda_buffers.t2s_curr;

    // nnf source and target already on device from minimize
    // only need dimensions for kernel launch
    int nnf_src_h = nnf.source_size().height;
    int nnf_src_w = nnf.source_size().width;
    int nnf_tgt_h = nnf.target_size().height;
    int nnf_tgt_w = nnf.target_size().width;

    // // handle upscaled source, different resolution, must upload separately
    // if (upscaled) {
    //     auto src_packed = nnf._pack_pixel_data(source, include_pixels);
    //     auto src_host = nnf._make_host_buffers(src_packed, source);
        
    //     int upscaled_pixels = source.size().height * source.size().width;
    //     cuda_bufs->upscaled_src_bufs.allocate_buffers(upscaled_pixels, has_gmask);
        
    //     cudaCheckError(cudaMemcpy(cuda_bufs->upscaled_src_bufs.data,
    //                               src_host.data,
    //                               upscaled_pixels * sizeof(PixelData),
    //                               cudaMemcpyHostToDevice));
    // }

    // // select correct source buffer for kernel
    // // upscaled: use upscaled_src_bufs
    // // non-upscaled: src_bufs already valid from minimize
    // PixelData *d_src = upscaled ? cuda_bufs->upscaled_src_bufs.data 
    //                             : cuda_bufs->src_bufs.data;
    // int src_h = upscaled ? source.size().height : nnf_src_h;
    // int src_w = upscaled ? source.size().width  : nnf_src_w;

	auto src_packed = nnf._pack_pixel_data(source, include_pixels);
    auto src_host   = nnf._make_host_buffers(src_packed, source);

    int src_pixels = source.size().height * source.size().width;
    cuda_bufs->upscaled_src_bufs.allocate_buffers(src_pixels, has_gmask);
    cudaCheckError(cudaMemcpy(cuda_bufs->upscaled_src_bufs.data,
                              src_host.data,
                              src_pixels * sizeof(PixelData),
                              cudaMemcpyHostToDevice));

    PixelData *d_src = cuda_bufs->upscaled_src_bufs.data;
    int src_h = source.size().height;
    int src_w = source.size().width;

    launch_expectation_step(cuda_bufs->d_vote, d_field_ptr, d_src, 
        cuda_bufs->src_bufs.data, cuda_bufs->tgt_bufs.data, 
        has_gmask, src_h, src_w, nnf_src_h, nnf_src_w,
        nnf_tgt_h, nnf_tgt_w, source2target, upscaled,
        m_distance_metric->patch_size());
	
	cudaMemcpy(vote.ptr<double>(0,0), m_cuda_buffers.d_vote,
					nnf_tgt_h * nnf_tgt_w * 
					4 * sizeof(double),
					cudaMemcpyDeviceToHost);
}

void Inpainting::_maximization_step_cuda(MaskedImage &target,
                                          CudaNNFDeviceBuffers *cuda_bufs, cv::Mat &vote) {
    int tgt_h = target.size().height;
    int tgt_w = target.size().width;
    int tgt_pixels = tgt_h * tgt_w;
    bool has_gmask = !target.global_mask().empty();

    cuda_bufs->ensure_new_target_buffer(tgt_pixels);

	cudaMemcpy(m_cuda_buffers.d_vote, vote.ptr<double>(0,0),
					tgt_h * tgt_w * 
					4 * sizeof(double),
					cudaMemcpyHostToDevice);
    cudaCheckError(cudaMemcpy(cuda_bufs->d_new_target_img,
                              target.image().ptr<unsigned char>(0, 0),
                              tgt_pixels * 3, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(cuda_bufs->d_new_target_mask,
                              target.mask().ptr<unsigned char>(0, 0),
                              tgt_pixels, cudaMemcpyHostToDevice));

    const unsigned char *d_gmask = nullptr;
    // gmask doesn't change so we can reuse tgt_bufs gmask if available
    // for now upload separately if needed
    unsigned char *d_gmask_buf = nullptr;
    if (has_gmask) {
        cudaMalloc(&d_gmask_buf, tgt_pixels);
        cudaMemcpy(d_gmask_buf,
                   target.global_mask().ptr<unsigned char>(0, 0),
                   tgt_pixels, cudaMemcpyHostToDevice);
        d_gmask = d_gmask_buf;
    }

    launch_maximization_step(cuda_bufs->d_new_target_img,
        cuda_bufs->d_new_target_mask, d_gmask,
        cuda_bufs->d_vote, has_gmask, tgt_h, tgt_w);

    cudaCheckError(cudaMemcpy(target.get_mutable_image(0, 0),
                              cuda_bufs->d_new_target_img,
                              tgt_pixels * 3, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(target.get_mutable_mask(0, 0),
                              cuda_bufs->d_new_target_mask,
                              tgt_pixels, cudaMemcpyDeviceToHost));

    if (d_gmask_buf) cudaFree(d_gmask_buf);
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