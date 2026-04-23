#pragma once
#include "cuda/cuda_buffers.h"
#include "patchmatch/masked_image.h"
#include <opencv2/core.hpp>

bool checkGpuCandidacy(cv::Size size);

class PatchDistanceMetric {
  public:
	PatchDistanceMetric(int patch_size) : m_patch_size(patch_size) {}
	virtual ~PatchDistanceMetric() = default;

	inline int patch_size() const { return m_patch_size; }
	virtual int operator()(const MaskedImage &source, int source_y,
						   int source_x, const MaskedImage &target,
						   int target_y, int target_x) const = 0;
	static const int kDistanceScale;

  protected:
	int m_patch_size;
};

class NearestNeighborField {
  public:
	NearestNeighborField()
		: m_source(), m_target(), m_field(), m_distance_metric(nullptr) {}
	NearestNeighborField(const MaskedImage &source, const MaskedImage &target,
						 const PatchDistanceMetric *metric, int max_retry = 20,
						 bool skip_host_init = false)
		: m_source(source), m_target(target), m_distance_metric(metric) {
		m_field = cv::Mat(m_source.size(), CV_32SC3);
		if (!skip_host_init)
			_randomize_field(max_retry);
	}
	NearestNeighborField(const MaskedImage &source, const MaskedImage &target,
						 const PatchDistanceMetric *metric,
						 const NearestNeighborField &other, int max_retry = 20,
						 bool skip_host_init = false)
		: m_source(source), m_target(target), m_distance_metric(metric) {
		m_field = cv::Mat(m_source.size(), CV_32SC3);
		if (!skip_host_init)
			_initialize_field_from(other, max_retry);
	}

	const MaskedImage &source() const { return m_source; }
	const MaskedImage &target() const { return m_target; }
	inline cv::Size source_size() const { return m_source.size(); }
	inline cv::Size target_size() const { return m_target.size(); }
	inline cv::Mat &mutable_field() { return m_field; }
	inline MaskedImage &mutable_source() { return m_source; }
	inline MaskedImage &mutable_target() { return m_target; }
	inline void set_source(const MaskedImage &source) { m_source = source; }
	inline void set_target(const MaskedImage &target) { m_target = target; }

	inline int *mutable_ptr(int y, int x) { return m_field.ptr<int>(y, x); }
	inline const int *ptr(int y, int x) const { return m_field.ptr<int>(y, x); }

	inline int at(int y, int x, int c) const {
		return m_field.ptr<int>(y, x)[c];
	}
	inline int &at(int y, int x, int c) { return m_field.ptr<int>(y, x)[c]; }
	inline void set_identity(int y, int x) {
		auto ptr = mutable_ptr(y, x);
		ptr[0] = y, ptr[1] = x, ptr[2] = 0;
	}

	void minimize(int nr_pass, bool is_gpu_enabled,
				  CudaNNFDeviceBuffers *cuda_bufs = nullptr,
				  int *d_field_ptr = nullptr, int *d_field_scratch = nullptr);
	void minimize_cuda(int nr_pass, CudaNNFDeviceBuffers *bufs,
					   int *d_field_ptr, int *device_field_scratch);
	void initialize_cuda_randomize(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
								   int max_retry, unsigned int seed);
	void initialize_cuda_from(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
							  const int *other_d_field_ptr,
							  cv::Size other_source_size, int max_retry,
							  unsigned int seed);
	void set_identity_cuda(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
						   const MaskedImage &mask_source);

  private:
	inline int _distance(int source_y, int source_x, int target_y,
						 int target_x) {
		return (*m_distance_metric)(m_source, source_y, source_x, m_target,
									target_y, target_x);
	}

	void _randomize_field(int max_retry = 20, bool reset = true);
	void _initialize_field_from(const NearestNeighborField &other,
								int max_retry);
	void _minimize_link(int y, int x, int direction);
	HostImageBuffers _make_host_buffers(std::vector<PixelData> &packed,
										const MaskedImage &img) const;
	std::vector<PixelData> _pack_pixel_data(const MaskedImage &img,
											bool include_pixels) const;

	MaskedImage m_source;
	MaskedImage m_target;
	cv::Mat m_field; // { y_target, x_target, distance_scaled }
	const PatchDistanceMetric *m_distance_metric;
};

class PatchSSDDistanceMetric : public PatchDistanceMetric {
  public:
	using PatchDistanceMetric::PatchDistanceMetric;
	virtual int operator()(const MaskedImage &source, int source_y,
						   int source_x, const MaskedImage &target,
						   int target_y, int target_x) const;
	static const int kSSDScale;
};