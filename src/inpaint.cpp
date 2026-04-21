#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../include/CycleTimer.h"
#include "../include/cuda_helpers.h"
#include "../include/inpaint.h"
#include "masked_image.h"
#include <iostream>

#include <omp.h>
#include <vector>

#define MAX_RETRIES 20

static int thr_count;
namespace {
// lookup table that converts patch distance to similarity
static std::vector<double> kDistance2Similarity;

void init_kDistance2Similarity() {
	double base[11] = {1.0,	 0.99,	0.96,	0.83,	0.38, 0.11,
					   0.02, 0.005, 0.0006, 0.0001, 0};
	int length = (PatchDistanceMetric::kDistanceScale + 1);
	kDistance2Similarity.resize(length);
	for (int i = 0; i < length; ++i) {
		double t = (double)i / length;
		int j = (int)(100 * t);
		int k = j + 1;
		double vj = (j < 11) ? base[j] : 0;
		double vk = (k < 11) ? base[k] : 0;
		kDistance2Similarity[i] = vj + (100 * t - j) * (vk - vj);
	}
}

inline void _weighted_copy(const MaskedImage &source, int ys, int xs,
						   cv::Mat &target, int yt, int xt, double weight) {
	if (source.is_masked(ys, xs))
		return;
	if (source.is_globally_masked(ys, xs))
		return;

	auto source_ptr = source.get_image(ys, xs);
	auto target_ptr = target.ptr<double>(yt, xt);

#pragma unroll
	for (int c = 0; c < 3; ++c)
		target_ptr[c] += static_cast<double>(source_ptr[c]) * weight;
	target_ptr[3] += weight;
}
} // namespace

/**
 * This algorithm uses a version proposed by Xavier Philippeau.
 */

Inpainting::Inpainting(cv::Mat image, cv::Mat mask,
					   const PatchDistanceMetric *metric, bool is_gpu_enabled)
	: m_initial(image, mask), m_distance_metric(metric), m_pyramid(),
	  m_source2target(), m_target2source(), m_gpu_enabled(is_gpu_enabled) {
	_initialize_pyramid();
}

Inpainting::Inpainting(cv::Mat image, cv::Mat mask, cv::Mat global_mask,
					   const PatchDistanceMetric *metric, bool is_gpu_enabled)
	: m_initial(image, mask, global_mask), m_distance_metric(metric),
	  m_pyramid(), m_source2target(), m_target2source(),
	  m_gpu_enabled(is_gpu_enabled) {
	_initialize_pyramid();
}

/**
 * Sets up the multi-resolution pyramid by repeatedly downsampling the initial
 * image until it is smaller than the patch size. The pyramid is ordered from
 * higher resolution (index 0) to coarser resolution (last index). Also
 * initializes the distance-to-similarity lookup table if not already done.
 */
void Inpainting::_initialize_pyramid() {
	MaskedImage source = m_initial;
	m_pyramid.push_back(source);
	while (source.size().height > m_distance_metric->patch_size() &&
		   source.size().width > m_distance_metric->patch_size()) {
		source = source.downsample();
		m_pyramid.push_back(source);
	}

	if (kDistance2Similarity.size() == 0) {
		init_kDistance2Similarity();
	}
}

cv::Mat Inpainting::run(bool verbose, bool verbose_visualize,
						unsigned int random_seed) {
	srand(random_seed);
	const int nr_levels = m_pyramid.size();
	thr_count = m_gpu_enabled ? omp_get_max_threads() : 1;

	MaskedImage source, target;
	cv::Size prev_source_size, prev_target_size;
	auto prevEolTime = startTime;
	for (int level = nr_levels - 1; level >= 0; --level) {
		if (verbose)
			std::cout << "Inpainting level: " << level << std::endl;

		source = m_pyramid[level];

		if (level == nr_levels - 1) {
			target = source.clone();
			target.clear_mask();
			m_source2target =
				NearestNeighborField(source, target, m_distance_metric, MAX_RETRIES, m_gpu_enabled);
			m_target2source =
				NearestNeighborField(target, source, m_distance_metric, MAX_RETRIES, m_gpu_enabled);
		} else {
			m_source2target = NearestNeighborField(
				source, target, m_distance_metric, m_source2target, MAX_RETRIES, m_gpu_enabled);
			m_target2source = NearestNeighborField(
				target, source, m_distance_metric, m_target2source, MAX_RETRIES, m_gpu_enabled);
		}

		if (m_gpu_enabled) {
			auto src_size = source.size();
			auto tgt_size = target.size();
			bool has_gmask = !source.global_mask().empty();

			m_cuda_buffers.allocate_device_buffers(
				src_size.height * src_size.width,
				tgt_size.height * tgt_size.width,
				has_gmask);

			unsigned int init_seed = (unsigned int)rand();

			if (level == nr_levels - 1) {
				m_source2target.initialize_cuda_randomize(
					&m_cuda_buffers, m_cuda_buffers.s2t_curr, 20, init_seed);
				m_target2source.initialize_cuda_randomize(
					&m_cuda_buffers, m_cuda_buffers.t2s_curr, 20, init_seed ^ 0xDEADBEEF);
			} else {
				// prev buffers still hold the previous level's field.
				// s2t's "other source" is the previous level's source.
				// t2s's "other source" is the previous level's target.
				m_source2target.initialize_cuda_from(
					&m_cuda_buffers, m_cuda_buffers.s2t_curr,
					m_cuda_buffers.s2t_prev,
					prev_source_size, 20, init_seed);
				m_target2source.initialize_cuda_from(
					&m_cuda_buffers, m_cuda_buffers.t2s_curr,
					m_cuda_buffers.t2s_prev,
					prev_target_size, 20, init_seed ^ 0xABCDu);
			}
		}

		prev_source_size = source.size();
    	prev_target_size = target.size();


		auto postInitTime = CycleTimer::currentSeconds();

		if (verbose)
			std::cout << "Initialization done." << std::endl;

		if (verbose_visualize) {
			auto visualize_size = m_initial.size();
			cv::Mat source_visualize(visualize_size, m_initial.image().type());
			cv::resize(source.image(), source_visualize, visualize_size);
			cv::imshow("Source", source_visualize);
			cv::Mat target_visualize(visualize_size, m_initial.image().type());
			cv::resize(target.image(), target_visualize, visualize_size);
			cv::imshow("Target", target_visualize);
			std::cout << "Press a key to continue" << std::endl;
			cv::waitKey(0);
		}

		target = _expectation_maximization(source, target, level, verbose);

		 if (m_gpu_enabled) {
			m_cuda_buffers.swap_s2t_fields();
			m_cuda_buffers.swap_t2s_fields();
		}
		auto eolTime = CycleTimer::currentSeconds();
		LOG("time to process level %d: %lfs, initTime: %lfs\n", level, eolTime - prevEolTime, postInitTime - prevEolTime);
		prevEolTime = eolTime;
	}

	return target.image();
}

// EM-Like algorithm (see "PatchMatch" - page 6).
// Returns a double sized target image (unless level = 0).
MaskedImage Inpainting::_expectation_maximization(MaskedImage source,
												  MaskedImage target, int level,
												  bool verbose) {
	const int nr_iters_em = 1 + 2 * level;
	// coarser levels require lesser iterations to converge
	const int nr_iters_nnf = static_cast<int>(std::min(7, 1 + level));
	const int patch_size = m_distance_metric->patch_size();

	MaskedImage new_source, new_target;

	int gpu_nnf_iters = std::max(1, nr_iters_nnf / 4);
	LOG("level %d, num_iters %d\n", level, m_gpu_enabled ? gpu_nnf_iters : nr_iters_em );
	for (int iter_em = 0; iter_em < nr_iters_em; ++iter_em) {
		double t_start = CycleTimer::currentSeconds();
		if (iter_em != 0) {
			m_source2target.set_target(new_target);
			m_target2source.set_source(new_target);
			target = new_target;
		}

		if (verbose)
			std::cout << "EM Iteration: " << iter_em << std::endl;

		auto size = source.size();
		// bool is_gpu_candidate = checkGpuCandidacy(size);
		// iterates every pixel and checks if
		// the patch centered at i,j overlaps
		// with the mask. If not, sets itself
		// as nearest patch (identity)
		if (m_gpu_enabled) {
			// allocate device buffers first if not done.
			if (iter_em == 0) {
				auto src_size = source.size();
				auto tgt_size = target.size();
				m_cuda_buffers.allocate_device_buffers(
					src_size.height * src_size.width,
					tgt_size.height * tgt_size.width,
					!source.global_mask().empty());
			}
			m_source2target.set_identity_cuda(&m_cuda_buffers,
											m_cuda_buffers.s2t_curr, source);
			m_target2source.set_identity_cuda(&m_cuda_buffers,
											m_cuda_buffers.t2s_curr, source);
		} else {
			for (int i = 0; i < size.height; ++i) {
				for (int j = 0; j < size.width; ++j) {
					if (!source.contains_mask(i, j, patch_size)) {
						m_source2target.set_identity(i, j);
						m_target2source.set_identity(i, j);
					}
				}
			}
		}
		if (verbose)
			std::cout << "  NNF minimization started." << std::endl;
		if (m_gpu_enabled) {
			// if (iter_em == 0) {
			// 	auto src_size = source.size();
			// 	auto tgt_size = target.size();
			// 	m_cuda_buffers.allocate_device_buffers(
			// 		src_size.height * src_size.width,
			// 		tgt_size.height * tgt_size.width,
			// 		!source.global_mask().empty());
			// }
			m_source2target.minimize(gpu_nnf_iters, true, &m_cuda_buffers, m_cuda_buffers.s2t_curr, m_cuda_buffers.s2t_prev);
			m_target2source.minimize(gpu_nnf_iters, true, &m_cuda_buffers, m_cuda_buffers.t2s_curr, m_cuda_buffers.t2s_prev);
		} else {
			m_source2target.minimize(nr_iters_nnf, false);
			m_target2source.minimize(nr_iters_nnf, false);
		}

		cuda_device_sync();

		double t_after_nnf_minimize = CycleTimer::currentSeconds();
		if (verbose)
			std::cout << "  NNF minimization finished." << std::endl;

		// Instead of upsizing the final target, we build the last target from
		// the next level source image. Thus, the final target is less blurry
		// (see "Space-Time Video Completion" - page 5).
		bool upscaled = false;
		if (level >= 1 && iter_em == nr_iters_em - 1) {
			new_source = m_pyramid[level - 1];
			new_target = target.upsample(new_source.size().width,
										 new_source.size().height,
										 m_pyramid[level - 1].global_mask());
			upscaled = true;
		} else {
			new_source = m_pyramid[level];
			new_target = target.clone();
		}

		double t_after_upscaling = CycleTimer::currentSeconds();

		auto vote = cv::Mat(new_target.size(), CV_64FC4);
		vote.setTo(cv::Scalar::all(0));

		// E step - Votes for best patch from NNF Source->Target (completeness)
		// and Target->Source (coherence).

		// Completeness - ensures that the output image contains as much
		// information as possible from the input as possible
		_expectation_step(m_source2target, 1, vote, new_source, upscaled,
						  m_gpu_enabled);
		if (verbose)
			std::cout << "  Expectation source to target finished."
					  << std::endl;

		// Coherence - ensures that the output is coherent wrt the input and
		// that new visual structures are penalised
		_expectation_step(m_target2source, 0, vote, new_source, upscaled,
						  m_gpu_enabled);
		if (verbose)
			std::cout << "  Expectation target to source finished."
					  << std::endl;

		double t_after_estep = CycleTimer::currentSeconds();

		// M step - Compile votes (averaged) and update pixel values.
		_maximization_step(new_target, vote, m_gpu_enabled);
		if (verbose)
			std::cout << "  Minimization step finished." << std::endl;

		double t_after_em = CycleTimer::currentSeconds();
		LOG("[EM level=%d iter=%d] nnf=%.3fs upscaling=%.3fs expectation=%.3fs "
			"maximization=%.3fs, total=%.3fs\n",
			level, iter_em, t_after_nnf_minimize - t_start,
			t_after_upscaling - t_after_nnf_minimize,
			t_after_estep - t_after_upscaling, t_after_em - t_after_estep, t_after_em - t_start);
	}

	return new_target;
}

// Expectation step: vote for best estimations of each pixel.
// Patch voting
void Inpainting::_expectation_step(const NearestNeighborField &nnf,
								   bool source2target, cv::Mat &vote,
								   const MaskedImage &source, bool upscaled,
								   bool is_parallel) {
	auto source_size = nnf.source_size();
	auto target_size = nnf.target_size();
	const int patch_size = m_distance_metric->patch_size();

	std::vector<cv::Mat> local_votes(thr_count);
	for (int t = 0; t < thr_count; ++t) {
		local_votes[t] = cv::Mat::zeros(vote.size(), vote.type());
	}

#pragma omp parallel num_threads(thr_count)
	{
		int tid = omp_get_thread_num();
		auto local_vote = local_votes[tid];

		#pragma omp for collapse(2) schedule(static)
		for (int i = 0; i < source_size.height; ++i) {
			for (int j = 0; j < source_size.width; ++j) {
				if (nnf.source().is_globally_masked(i, j))
					continue;
				int yp = nnf.at(i, j, 0), xp = nnf.at(i, j, 1),
					dp = nnf.at(i, j, 2);
				double w = kDistance2Similarity[dp];

				for (int di = -patch_size; di <= patch_size; ++di) {
					for (int dj = -patch_size; dj <= patch_size; ++dj) {
						int ys = i + di, xs = j + dj, yt = yp + di,
							xt = xp + dj;
						if (!(ys >= 0 && ys < source_size.height && xs >= 0 &&
							  xs < source_size.width))
							continue;
						if (nnf.source().is_globally_masked(ys, xs))
							continue;
						if (!(yt >= 0 && yt < target_size.height && xt >= 0 &&
							  xt < target_size.width))
							continue;
						if (nnf.target().is_globally_masked(yt, xt))
							continue;

						if (!source2target) {
							std::swap(ys, yt);
							std::swap(xs, xt);
						}

						if (upscaled) {
							for (int uy = 0; uy < 2; ++uy) {
								for (int ux = 0; ux < 2; ++ux) {
									_weighted_copy(source, 2 * ys + uy,
												   2 * xs + ux, local_vote,
												   2 * yt + uy, 2 * xt + ux, w);
								}
							}
						} else {
							_weighted_copy(source, ys, xs, local_vote, yt, xt,
										   w);
						}
					}
				}
			}
		}
	}

	vote.setTo(cv::Scalar::all(0));

	#pragma omp parallel for collapse(2) schedule(static) num_threads(thr_count)
	for (int y = 0; y < vote.rows; ++y) {
		for (int x = 0; x < vote.cols; ++x) {
			double *dst = vote.ptr<double>(y, x);
			for (int t = 0; t < thr_count; ++t) {
				const double *src = local_votes[t].ptr<double>(y, x);
				dst[0] += src[0];
				dst[1] += src[1];
				dst[2] += src[2];
				dst[3] += src[3];
			}
		}
	}
}

// Maximization Step: maximum likelihood of target pixel.
void Inpainting::_maximization_step(MaskedImage &target, const cv::Mat &vote, bool is_parallel) {
	auto target_size = target.size();
	int thr_count = is_parallel ? std::min(omp_get_max_threads(), 16) : 1;

	#pragma omp parallel for collapse(2) num_threads(thr_count) schedule(static)
	for (int i = 0; i < target_size.height; ++i) {
		for (int j = 0; j < target_size.width; ++j) {
			const double *source_ptr = vote.ptr<double>(i, j);
			unsigned char *target_ptr = target.get_mutable_image(i, j);

			if (target.is_globally_masked(i, j)) {
				continue;
			}

			if (source_ptr[3] > 0) {
				unsigned char r = cv::saturate_cast<unsigned char>(
					source_ptr[0] / source_ptr[3]);
				unsigned char g = cv::saturate_cast<unsigned char>(
					source_ptr[1] / source_ptr[3]);
				unsigned char b = cv::saturate_cast<unsigned char>(
					source_ptr[2] / source_ptr[3]);
				target_ptr[0] = r, target_ptr[1] = g, target_ptr[2] = b;
			} else {
				target.set_mask(i, j, 0);
			}
		}
	}
}
