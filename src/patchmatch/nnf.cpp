#include <algorithm>
#include <cmath>

#include "patchmatch/masked_image.h"
#include "patchmatch/nnf.h"

/**
 * Nearest-Neighbor Field (see PatchMatch algorithm).
 * This algorithme uses a version proposed by Xavier Philippeau.
 *
 */

template <typename T> T clamp(T value, T min_value, T max_value) {
	return std::min(std::max(value, min_value), max_value);
}

void NearestNeighborField::_randomize_field(int max_retry, bool reset) {
	auto this_size = source_size();
	for (int i = 0; i < this_size.height; ++i) {
		for (int j = 0; j < this_size.width; ++j) {
			if (m_source.is_globally_masked(i, j))
				continue;

			auto this_ptr = mutable_ptr(i, j);
			int distance =
				reset ? PatchDistanceMetric::kDistanceScale : this_ptr[2];
			if (distance < PatchDistanceMetric::kDistanceScale) {
				continue;
			}

			int i_target = 0, j_target = 0;
			for (int t = 0; t < max_retry; ++t) {
				i_target = rand() % this_size.height;
				j_target = rand() % this_size.width;
				if (m_target.is_globally_masked(i_target, j_target))
					continue;

				distance = _distance(i, j, i_target, j_target);
				if (distance < PatchDistanceMetric::kDistanceScale)
					break;
			}

			this_ptr[0] = i_target, this_ptr[1] = j_target,
			this_ptr[2] = distance;
		}
	}
}

void NearestNeighborField::_initialize_field_from(
	const NearestNeighborField &other, int max_retry) {
	const auto &this_size = source_size();
	const auto &other_size = other.source_size();
	double fi = static_cast<double>(this_size.height) / other_size.height;
	double fj = static_cast<double>(this_size.width) / other_size.width;

	for (int i = 0; i < this_size.height; ++i) {
		for (int j = 0; j < this_size.width; ++j) {
			if (m_source.is_globally_masked(i, j))
				continue;

			int ilow = static_cast<int>(
				std::min(i / fi, static_cast<double>(other_size.height - 1)));
			int jlow = static_cast<int>(
				std::min(j / fj, static_cast<double>(other_size.width - 1)));
			auto this_value = mutable_ptr(i, j);
			auto other_value = other.ptr(ilow, jlow);

			this_value[0] = static_cast<int>(other_value[0] * fi);
			this_value[1] = static_cast<int>(other_value[1] * fj);
			this_value[2] = _distance(i, j, this_value[0], this_value[1]);
		}
	}

	_randomize_field(max_retry, false);
}

bool checkGpuCandidacy(cv::Size size) {
	int h = size.height;
	int w = size.width;

	int num_threads = 256;
	int blocks = ((h * w) + num_threads - 1) / num_threads;

	return blocks > 1;
}

void NearestNeighborField::minimize(int nr_pass, bool is_gpu_enabled, CudaNNFDeviceBuffers *cuda_bufs, int4 *d_field_ptr, int4 *d_field_scratch) {
	const auto &this_size = source_size();

	if (is_gpu_enabled) {
		minimize_cuda(nr_pass, cuda_bufs, d_field_ptr, d_field_scratch);
		return;
	}

	while (nr_pass--) {
		// top left to bottom right
		for (int i = 0; i < this_size.height; ++i)
			for (int j = 0; j < this_size.width; ++j) {
				if (m_source.is_globally_masked(i, j))
					continue;
				// checks channel 2 - the distance
				// score for the current best match
				// at pixel (i,j). If 0, then best
				// match is found
				if (at(i, j, 2) > 0)
					_minimize_link(i, j, +1);
			}
		// bottom right to top left
		for (int i = this_size.height - 1; i >= 0; --i)
			for (int j = this_size.width - 1; j >= 0; --j) {
				if (m_source.is_globally_masked(i, j))
					continue;
				if (at(i, j, 2) > 0)
					_minimize_link(i, j, -1);
			}
	}
}

// Single Iteration. Section 3.2 in the PatchMatch paper
void NearestNeighborField::_minimize_link(int y, int x, int direction) {
	const auto &this_size = source_size();
	const auto &this_target_size = target_size();
	auto this_ptr = mutable_ptr(y, x);

	// propagation along the y direction.
	if (y - direction >= 0 && y - direction < this_size.height &&
		!m_source.is_globally_masked(y - direction, x)) {
		int yp = at(y - direction, x, 0) + direction;
		int xp = at(y - direction, x, 1);
		int dp = _distance(y, x, yp, xp);
		if (dp < at(y, x, 2)) {
			this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
		}
	}

	// propagation along the x direction.
	if (x - direction >= 0 && x - direction < this_size.width &&
		!m_source.is_globally_masked(y, x - direction)) {
		int yp = at(y, x - direction, 0);
		int xp = at(y, x - direction, 1) + direction;
		int dp = _distance(y, x, yp, xp);
		if (dp < at(y, x, 2)) {
			this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
		}
	}

	// random search with a progressive step size.
	int random_scale =
		(std::min(this_target_size.height, this_target_size.width) - 1) / 2;
	while (random_scale > 0) {
		int yp = this_ptr[0] + (rand() % (2 * random_scale + 1) - random_scale);
		int xp = this_ptr[1] + (rand() % (2 * random_scale + 1) - random_scale);
		yp = clamp(yp, 0, target_size().height - 1);
		xp = clamp(xp, 0, target_size().width - 1);

		if (m_target.is_globally_masked(yp, xp)) {
			random_scale /= 2;
		}

		int dp = _distance(y, x, yp, xp);
		if (dp < at(y, x, 2)) {
			this_ptr[0] = yp, this_ptr[1] = xp, this_ptr[2] = dp;
		}
		random_scale /= 2;
	}
}

const int PatchDistanceMetric::kDistanceScale = 65535;
const int PatchSSDDistanceMetric::kSSDScale = 9 * 255 * 255;

namespace {

inline int pow2(int i) { return i * i; }

int distance_masked_images(const MaskedImage &source, int ys, int xs,
						   const MaskedImage &target, int yt, int xt,
						   int patch_size) {
	long double distance = 0;
	long double wsum = 0;

	source.compute_image_gradients();
	target.compute_image_gradients();

	auto source_size = source.size();
	auto target_size = target.size();

	for (int dy = -patch_size; dy <= patch_size; ++dy) {
		const int yys = ys + dy, yyt = yt + dy;

		if (yys <= 0 || yys >= source_size.height - 1 || yyt <= 0 ||
			yyt >= target_size.height - 1) {
			distance += (long double)(PatchSSDDistanceMetric::kSSDScale) *
						(2 * patch_size + 1);
			wsum += 2 * patch_size + 1;
			continue;
		}

		const auto *p_si = source.image().ptr<unsigned char>(yys, 0);
		const auto *p_ti = target.image().ptr<unsigned char>(yyt, 0);
		const auto *p_sm = source.mask().ptr<unsigned char>(yys, 0);
		const auto *p_tm = target.mask().ptr<unsigned char>(yyt, 0);

		const unsigned char *p_sgm = nullptr;
		const unsigned char *p_tgm = nullptr;
		if (!source.global_mask().empty()) {
			p_sgm = source.global_mask().ptr<unsigned char>(yys, 0);
			p_tgm = target.global_mask().ptr<unsigned char>(yyt, 0);
		}

		const auto *p_sgy = source.grady().ptr<unsigned char>(yys, 0);
		const auto *p_tgy = target.grady().ptr<unsigned char>(yyt, 0);
		const auto *p_sgx = source.gradx().ptr<unsigned char>(yys, 0);
		const auto *p_tgx = target.gradx().ptr<unsigned char>(yyt, 0);

		for (int dx = -patch_size; dx <= patch_size; ++dx) {
			int xxs = xs + dx, xxt = xt + dx;
			wsum += 1;

			if (xxs <= 0 || xxs >= source_size.width - 1 || xxt <= 0 ||
				xxt >= target_size.width - 1) {
				distance += PatchSSDDistanceMetric::kSSDScale;
				continue;
			}

			if (p_sm[xxs] || p_tm[xxt] || (p_sgm && p_sgm[xxs]) ||
				(p_tgm && p_tgm[xxt])) {
				distance += PatchSSDDistanceMetric::kSSDScale;
				continue;
			}

			int ssd = 0;
			for (int c = 0; c < 3; ++c) {
				int s_value = p_si[xxs * 3 + c];
				int t_value = p_ti[xxt * 3 + c];
				int s_gy = p_sgy[xxs * 3 + c];
				int t_gy = p_tgy[xxt * 3 + c];
				int s_gx = p_sgx[xxs * 3 + c];
				int t_gx = p_tgx[xxt * 3 + c];

				ssd += pow2(static_cast<int>(s_value) - t_value);
				ssd += pow2(static_cast<int>(s_gx) - t_gx);
				ssd += pow2(static_cast<int>(s_gy) - t_gy);
			}
			distance += ssd;
		}
	}

	distance /= (long double)(PatchSSDDistanceMetric::kSSDScale);

	int res = int(PatchDistanceMetric::kDistanceScale * distance / wsum);
	if (res < 0 || res > PatchDistanceMetric::kDistanceScale)
		return PatchDistanceMetric::kDistanceScale;
	return res;
}

} // namespace

int PatchSSDDistanceMetric::operator()(const MaskedImage &source, int source_y,
									   int source_x, const MaskedImage &target,
									   int target_y, int target_x) const {
	return distance_masked_images(source, source_y, source_x, target, target_y,
								  target_x, m_patch_size);
}