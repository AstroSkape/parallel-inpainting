#include "masked_image.h"
#include "nnf.h"
#include <opencv2/opencv.hpp>

extern double startTime;
class Inpainting {
  public:
	Inpainting(cv::Mat image, cv::Mat mask, const PatchDistanceMetric *metric,
			   bool is_gpu_enabled);
	Inpainting(cv::Mat image, cv::Mat mask, cv::Mat global_mask,
			   const PatchDistanceMetric *metric, bool is_gpu_enabled);
	cv::Mat run(bool verbose = false, bool verbose_visualize = false,
				unsigned int random_seed = 1212);

  private:
	void _initialize_pyramid(void);
	MaskedImage _expectation_maximization(MaskedImage source,
										  MaskedImage target, int level,
										  bool verbose);
	void _expectation_step(const NearestNeighborField &nnf, bool source2target,
						   cv::Mat &vote, const MaskedImage &source,
						   bool upscaled, bool is_parallel);
	void _maximization_step(MaskedImage &target, const cv::Mat &vote,
							bool is_parallel);

	void _create_fields_at_level(const MaskedImage &source,
								 const MaskedImage &target,
								 bool is_coarsest_level);
	void _initialize_fields_on_gpu(const MaskedImage &source,
								   const MaskedImage &target,
								   bool is_coarsest_level,
								   cv::Size prev_source_size,
								   cv::Size prev_target_size);
	void _visualize_runs(const MaskedImage &source,
						 const MaskedImage &target) const;
	void _set_identity_on_field(const MaskedImage &source,
								const MaskedImage &target, int iter_em,
								int patch_size);

	MaskedImage m_initial;
	std::vector<MaskedImage> m_pyramid;

	NearestNeighborField m_source2target;
	NearestNeighborField m_target2source;
	const PatchDistanceMetric *m_distance_metric;
	bool m_gpu_enabled;
	CudaNNFDeviceBuffers m_cuda_buffers;
};