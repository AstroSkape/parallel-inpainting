#include "cuda/cuda_buffers.h"
#include "cuda/cuda_helpers.cuh"
#include <cuda_runtime.h>
#include <iostream>

#define RED 0
#define BLACK 1

#define WANG_SALT 0x27d4eb2d

// Wang hash
// Code adapted from https://burtleburtle.net/bob/hash/integer.html
__device__ unsigned int wang_hash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * WANG_SALT;
	a = a ^ (a >> 15);
	return a;
}

__device__ int rand_range(unsigned int seed, int lo, int hi) {
	int range = hi - lo + 1;
	return lo + (int)(wang_hash(seed) % range);
}

__device__ int device_clamp(int val, int min_val, int max_val) {
	return min(max_val, max(val, min_val));
}

template <typename T> void realloc_device_ptr(T **ptr, size_t count) {
	if (*ptr) {
		cudaCheckError(cudaFree(*ptr));
	}
	cudaCheckError(cudaMalloc((void **)ptr, count * sizeof(T)));
}

/**
 * Allocates space only if needed. Max sized buffers are retained
 */
void DeviceImageBuffers::allocate_buffers(int num_pixels, bool has_gmask) {
	if (num_pixels > pixel_capacity) {
		realloc_device_ptr(&rgb_mask, num_pixels);
		realloc_device_ptr(&gx, num_pixels);
		realloc_device_ptr(&gy, num_pixels);
		pixel_capacity = num_pixels;
	}
}

void ensure_fields(int required_pixels, int &capacity, int **curr, int **prev) {
	if (required_pixels > capacity) {
		size_t count = required_pixels * 3;
		realloc_device_ptr(curr, count);
		realloc_device_ptr(prev, count);
		capacity = required_pixels;
	}
}

void CudaNNFDeviceBuffers::ensure_s2t_fields(int src_pixels) {
	ensure_fields(src_pixels, s2t_capacity, &s2t_curr, &s2t_prev);
}

void CudaNNFDeviceBuffers::ensure_t2s_fields(int tgt_pixels) {
	ensure_fields(tgt_pixels, t2s_capacity, &t2s_curr, &t2s_prev);
}

/**
 * Ensures that the buffers can hold the required number of pixels.
 * Reuses existing buffers to avoid repeated calls to malloc/free
 */
void CudaNNFDeviceBuffers::allocate_device_buffers(int src_pixels,
												   int tgt_pixels,
												   bool need_gmask) {
	src_bufs.allocate_buffers(src_pixels, need_gmask);
	tgt_bufs.allocate_buffers(tgt_pixels, need_gmask);
	ensure_s2t_fields(src_pixels);
	ensure_t2s_fields(tgt_pixels);
	ensure_vote(tgt_pixels);
}

void CudaNNFDeviceBuffers::ensure_vote(int tgt_pixels) {
	if (tgt_pixels > vote_capacity) {
		if (vote)
			cudaCheckError(cudaFree(vote));
		cudaCheckError(cudaMalloc(&vote, tgt_pixels * sizeof(float4)));
		vote_capacity = tgt_pixels;
	}
	cudaMemset(vote, 0, tgt_pixels * sizeof(float4));
}

void CudaNNFDeviceBuffers::upload_dist2sim(const double *host_table, int n) {
	// Convert host doubles to floats and upload once per run
	std::vector<float> tmp(n);
	for (int i = 0; i < n; ++i)
		tmp[i] = (float)host_table[i];
	if (!dist2sim)
		cudaCheckError(cudaMalloc(&dist2sim, n * sizeof(float)));
	cudaCheckError(cudaMemcpy(dist2sim, tmp.data(), n * sizeof(float),
							  cudaMemcpyHostToDevice));
}

void CudaNNFDeviceBuffers::init_streams() {
	if (!s2t_stream)
		cudaCheckError(cudaStreamCreate(&s2t_stream));
	if (!t2s_stream)
		cudaCheckError(cudaStreamCreate(&t2s_stream));
}

DeviceImageBuffers::~DeviceImageBuffers() {
	if (rgb_mask) {
		cudaCheckError(cudaFree(rgb_mask));
		cudaCheckError(cudaFree(gx));
		cudaCheckError(cudaFree(gy));
	}
}

CudaNNFDeviceBuffers::~CudaNNFDeviceBuffers() {
	if (s2t_curr)
		cudaCheckError(cudaFree(s2t_curr));
	if (s2t_prev)
		cudaCheckError(cudaFree(s2t_prev));
	if (t2s_curr)
		cudaCheckError(cudaFree(t2s_curr));
	if (t2s_prev)
		cudaCheckError(cudaFree(t2s_prev));
	if (vote)
		cudaCheckError(cudaFree(vote));
	if (dist2sim)
		cudaCheckError(cudaFree(dist2sim));
	if (s2t_stream)
		cudaCheckError(cudaStreamDestroy(s2t_stream));
	if (t2s_stream)
		cudaCheckError(cudaStreamDestroy(t2s_stream));
}

__device__ int compute_patch_dist(const uchar4 *src_rgb_mask,
								  const uchar4 *src_gx, const uchar4 *src_gy,
								  const uchar4 *tgt_rgb_mask,
								  const uchar4 *tgt_gx, const uchar4 *tgt_gy,
								  bool has_gmask, int ys, int xs, int yt,
								  int xt, int src_h, int src_w, int tgt_h,
								  int tgt_w, int patch_size, int best_d) {
	float distance = 0;
	const float wsum_total = (float)(2 * patch_size + 1) * (2 * patch_size + 1);
	const float kSSDScale = 9 * 255 * 255;
	const float kDistanceScale = 65535;

	// use for early termination
	const float max_distance =
		(float)(best_d + 1) * kSSDScale * wsum_total / kDistanceScale;

	for (int dy = -patch_size; dy <= patch_size; ++dy) {
		const int yys = ys + dy, yyt = yt + dy;

		if (yys <= 0 || yys >= src_h - 1 || yyt <= 0 || yyt >= tgt_h - 1) {
			distance += kSSDScale * (2 * patch_size + 1);
			continue;
		}

		for (int dx = -patch_size; dx <= patch_size; ++dx) {
			int xxs = xs + dx, xxt = xt + dx;

			if (xxs <= 0 || xxs >= src_w - 1 || xxt <= 0 || xxt >= tgt_w - 1) {
				distance += kSSDScale;
				continue;
			}

			int src_idx = yys * src_w + xxs;
			int tgt_idx = yyt * tgt_w + xxt;

			uchar4 src_rm = src_rgb_mask[src_idx]; // src rgb + mask + gmask
			uchar4 tgt_rm = tgt_rgb_mask[tgt_idx]; // tgt rgb + mask + gmask

			bool is_masked = (src_rm.w & 1) || (tgt_rm.w & 1) ||
							 (has_gmask && (src_rm.w & 2)) ||
							 (has_gmask && (tgt_rm.w & 2));
			if (is_masked) {
				distance += kSSDScale;
				continue;
			}

			uchar4 src_gradx = src_gx[src_idx];
			uchar4 tgt_gradx = tgt_gx[tgt_idx];
			uchar4 src_grady = src_gy[src_idx];
			uchar4 tgt_grady = tgt_gy[tgt_idx];

			int ssd = 0;

			// rgb channels
			ssd += (src_rm.x - tgt_rm.x) * (src_rm.x - tgt_rm.x);
			ssd += (src_rm.y - tgt_rm.y) * (src_rm.y - tgt_rm.y);
			ssd += (src_rm.z - tgt_rm.z) * (src_rm.z - tgt_rm.z);
			// gradx
			ssd += (src_gradx.x - tgt_gradx.x) * (src_gradx.x - tgt_gradx.x);
			ssd += (src_gradx.y - tgt_gradx.y) * (src_gradx.y - tgt_gradx.y);
			ssd += (src_gradx.z - tgt_gradx.z) * (src_gradx.z - tgt_gradx.z);
			// grad y
			ssd += (src_grady.x - tgt_grady.x) * (src_grady.x - tgt_grady.x);
			ssd += (tgt_grady.y - tgt_grady.y) * (src_grady.y - tgt_grady.y);
			ssd += (tgt_grady.z - tgt_grady.z) * (src_grady.z - tgt_grady.z);
			distance += ssd;
		}

		if (distance >= max_distance)
			return kDistanceScale;
	}

	distance /= kSSDScale;

	int res = (int)(kDistanceScale * distance / wsum_total);
	if (res < 0 || res > kDistanceScale)
		return kDistanceScale;
	return res;
}

/**
 * Copies the image host buffers to the device
 */
static void upload_image_buffers_to_device(DeviceImageBuffers &d_bufs,
										   const HostImageBuffers &h_bufs,
										   bool has_gmask,
										   cudaStream_t stream = nullptr) {
	int size = h_bufs.height * h_bufs.width;
	if (!stream) {
		cudaCheckError(cudaMemcpy(d_bufs.rgb_mask, h_bufs.rgb_mask,
								  size * sizeof(uchar4),
								  cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemcpy(d_bufs.gx, h_bufs.gx, size * sizeof(uchar4),
								  cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemcpy(d_bufs.gy, h_bufs.gy, size * sizeof(uchar4),
								  cudaMemcpyHostToDevice));
	} else {
		cudaCheckError(cudaMemcpyAsync(d_bufs.rgb_mask, h_bufs.rgb_mask,
									   size * sizeof(uchar4),
									   cudaMemcpyHostToDevice, stream));
		cudaCheckError(cudaMemcpyAsync(d_bufs.gx, h_bufs.gx,
									   size * sizeof(uchar4),
									   cudaMemcpyHostToDevice, stream));
		cudaCheckError(cudaMemcpyAsync(d_bufs.gy, h_bufs.gy,
									   size * sizeof(uchar4),
									   cudaMemcpyHostToDevice, stream));
	}
}

__global__ void
nnf_jump_flood_kernel(const int *field_in, int *field_out,
					  const uchar4 *src_rgb_mask, const uchar4 *src_gx,
					  const uchar4 *src_gy, const uchar4 *tgt_rgb_mask,
					  const uchar4 *tgt_gx, const uchar4 *tgt_gy,
					  bool has_gmask, int src_h, int src_w, int tgt_h,
					  int tgt_w, int patch_size, unsigned int seed, int jump) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int p_y = idx / src_w, p_x = idx % src_w;

	// Skip pixels that are globally masked in the source -- nothing to match.
	if (has_gmask && (src_rgb_mask[idx].w & 2))
		return;

	const int *nnf_in = field_in + idx * 3;
	int *nnf_out = field_out + idx * 3;

	int best_y = nnf_in[0];
	int best_x = nnf_in[1];
	int best_d = nnf_in[2];

	unsigned int step = 0;

	const int directions[8][2] = {{-jump, 0},	 {-jump, -jump}, {0, -jump},
								  {jump, -jump}, {jump, 0},		 {jump, jump},
								  {0, jump},	 {-jump, jump}};

	// propagate information at this jump distance
	for (int i = 0; i < 8; i++) {
		int newy = p_y + directions[i][0];
		int newx = p_x + directions[i][1];
		int newidx = newy * src_w + newx;

		if (newy < 0 || newy >= src_h || newx < 0 || newx >= src_w)
			continue;
		if (has_gmask && (src_rgb_mask[newidx].w & 2))
			continue;

		const int *neighbor_field = field_in + (newy * src_w + newx) * 3;
		int candidate_y =
			device_clamp(neighbor_field[0] - directions[i][0], 0, tgt_h - 1);
		int candidate_x =
			device_clamp(neighbor_field[1] - directions[i][1], 0, tgt_w - 1);

		int computed_dist = compute_patch_dist(
			src_rgb_mask, src_gx, src_gy, tgt_rgb_mask, tgt_gx, tgt_gy,
			has_gmask, p_y, p_x, candidate_y, candidate_x, src_h, src_w, tgt_h,
			tgt_w, patch_size, best_d);
		if (computed_dist < best_d) {
			best_x = candidate_x;
			best_y = candidate_y;
			best_d = computed_dist;
		}
	}

	// random search
	int random_scale = (min(tgt_h, tgt_w) - 1) / 2;
	while (random_scale > 0) {
		unsigned int s_base = seed + idx * 1337u + step * 7919u;
		int dy_off = rand_range(s_base, -random_scale, random_scale);
		int dx_off =
			rand_range(s_base + 2654435761u, -random_scale, random_scale);

		int yp = device_clamp(best_y + dy_off, 0, tgt_h - 1);
		int xp = device_clamp(best_x + dx_off, 0, tgt_w - 1);
		int idxp = yp * tgt_w + xp;

		if (!(has_gmask && (tgt_rgb_mask[idxp].w & 2))) {
			int d = compute_patch_dist(src_rgb_mask, src_gx, src_gy,
									   tgt_rgb_mask, tgt_gx, tgt_gy, has_gmask,
									   p_y, p_x, yp, xp, src_h, src_w, tgt_h,
									   tgt_w, patch_size, best_d);
			if (d < best_d) {
				best_y = yp;
				best_x = xp;
				best_d = d;
			}
		}

		random_scale >>= 1;
		step++;
	}

	nnf_out[0] = best_y;
	nnf_out[1] = best_x;
	nnf_out[2] = best_d;
}

extern "C" void launch_nnf_minimize(
	CudaNNFDeviceBuffers *bufs, int *d_field_ptr, int *d_field_scratch,
	int *h_field_ptr, const HostImageBuffers &src, const HostImageBuffers &tgt,
	bool has_gmask, int patch_size, int nr_pass, unsigned int random_seed) {

	int src_size = src.height * src.width;

	// Timing setup
	cudaEvent_t t0, t1, t2, t3;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);
	cudaEventCreate(&t2);
	cudaEventCreate(&t3);

	cudaEventRecord(t0);

	upload_image_buffers_to_device(bufs->src_bufs, src, has_gmask);
	upload_image_buffers_to_device(bufs->tgt_bufs, tgt, has_gmask);

	cudaEventRecord(t1);

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	const int jumps[6] = {8, 4, 2, 1, 2, 1};
	int *in_ptr = d_field_ptr;
	int *out_ptr = d_field_scratch;

	// LOG("[CUDA] Number of blocks: %d\n", blocks);

	for (int i = 0; i < nr_pass; i++) {
		for (int ji = 0; ji < 6; ji++) {
			unsigned int seed = random_seed + i * 12345u;

			nnf_jump_flood_kernel<<<blocks, num_threads>>>(
				in_ptr, out_ptr, bufs->src_bufs.rgb_mask, bufs->src_bufs.gx,
				bufs->src_bufs.gy, bufs->tgt_bufs.rgb_mask, bufs->tgt_bufs.gx,
				bufs->tgt_bufs.gy, has_gmask, src.height, src.width, tgt.height,
				tgt.width, patch_size, seed, jumps[ji]);

			// swap buffers
			int *tmp = in_ptr;
			in_ptr = out_ptr;
			out_ptr = tmp;
		}
	}

	if (in_ptr != d_field_ptr) {
		cudaCheckError(cudaMemcpy(d_field_ptr, in_ptr,
								  src_size * 3 * sizeof(int),
								  cudaMemcpyDeviceToDevice));
	}

	cudaEventRecord(t2);

	// copy back from device to host
	cudaCheckError(cudaMemcpy(h_field_ptr, d_field_ptr,
							  src_size * 3 * sizeof(int),
							  cudaMemcpyDeviceToHost));

	cudaEventRecord(t3);
	cudaEventSynchronize(t3);

	float h2d_ms, kernel_ms, d2h_ms;
	cudaEventElapsedTime(&h2d_ms, t0, t1);
	cudaEventElapsedTime(&kernel_ms, t1, t2);
	cudaEventElapsedTime(&d2h_ms, t2, t3);
	LOG("[TIMING] h2d=%.2fms kernel=%.2fms d2h=%.2fms total=%.2fms\n", h2d_ms,
		kernel_ms, d2h_ms, h2d_ms + kernel_ms + d2h_ms);

	cudaEventDestroy(t0);
	cudaEventDestroy(t1);
	cudaEventDestroy(t2);
	cudaEventDestroy(t3);
}

/**
 * Randomize field entries. If reset is false, touch pixels whose
 * current distance >= kDistanceScale. Algorithm taken directly from the
 * serial implementation
 */
__global__ void nnf_randomize_kernel(
	int *field, const uchar4 *src_rgb_mask, const uchar4 *src_gx,
	const uchar4 *src_gy, const uchar4 *tgt_rgb_mask, const uchar4 *tgt_gx,
	const uchar4 *tgt_gy, bool has_gmask, int src_h, int src_w, int tgt_h,
	int tgt_w, int patch_size, int max_retry, bool reset, unsigned int seed) {

	const int kDistanceScale = 65535;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int p_y = idx / src_w, p_x = idx % src_w; // pixel coordinates

	if (has_gmask && (src_rgb_mask[idx].w & 2))
		return;
	int *nnf_field = field + idx * 3; // 3 channels

	int dist = reset ? kDistanceScale : nnf_field[2];
	if (dist < kDistanceScale)
		return;

	int best_y = 0, best_x = 0, best_d = kDistanceScale;

	for (int t = 0; t < max_retry; t++) {
		unsigned int s = seed ^ (idx * 0x9E3779B1u) ^ (t * 0x85EBCA6Bu);
		int y_t = (int)(wang_hash(s) % (unsigned)tgt_h);
		int x_t = (int)(wang_hash(s + 2u) % (unsigned)tgt_w);

		if (has_gmask && (tgt_rgb_mask[y_t * tgt_w + x_t].w & 2))
			continue;

		int d =
			compute_patch_dist(src_rgb_mask, src_gx, src_gy, tgt_rgb_mask,
							   tgt_gx, tgt_gy, has_gmask, p_y, p_x, y_t, x_t,
							   src_h, src_w, tgt_h, tgt_w, patch_size, best_d);

		if (d < best_d) {
			best_y = y_t;
			best_x = x_t;
			best_d = d;
		}
		if (d < kDistanceScale)
			break;
	}

	nnf_field[0] = best_y;
	nnf_field[1] = best_x;
	nnf_field[2] = best_d;
}

/**
 * Lookup from previous level's field. Identical to the
 * _initialize_field_from on the CPU
 */
__global__ void nnf_initialize_from_kernel(
	int *field, const int *other_field, const uchar4 *src_rgb_mask,
	const uchar4 *src_gx, const uchar4 *src_gy, const uchar4 *tgt_rgb_mask,
	const uchar4 *tgt_gx, const uchar4 *tgt_gy, bool has_gmask, int src_h,
	int src_w, int tgt_h, int tgt_w, int other_src_h, int other_src_w,
	int patch_size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int i = idx / src_w, j = idx % src_w; // pixel coordinates

	if (has_gmask && (src_rgb_mask[idx].w & 2))
		return;

	float fi = (float)src_h / (float)other_src_h;
	float fj = (float)src_w / (float)other_src_w;

	int ilow = (int)fminf((float)i / fi, (float)(other_src_h - 1));
	int jlow = (int)fminf((float)j / fj, (float)(other_src_w - 1));

	const int *other_value = other_field + (ilow * other_src_w + jlow) * 3;

	int y_t = (int)((float)other_value[0] * fi);
	int x_t = (int)((float)other_value[1] * fj);

	y_t = device_clamp(y_t, 0, tgt_h - 1);
	x_t = device_clamp(x_t, 0, tgt_w - 1);

	int dist = compute_patch_dist(src_rgb_mask, src_gx, src_gy, tgt_rgb_mask,
								  tgt_gx, tgt_gy, has_gmask, i, j, y_t, x_t,
								  src_h, src_w, tgt_h, tgt_w, patch_size, 0);

	int *nnf_field = field + idx * 3;
	nnf_field[0] = y_t;
	nnf_field[1] = x_t;
	nnf_field[2] = dist;
}

extern "C" void
launch_nnf_randomize(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
					 const HostImageBuffers &src, const HostImageBuffers &tgt,
					 bool has_gmask, int patch_size, int max_retry, bool reset,
					 unsigned int seed, cudaStream_t stream) {

	int src_size = src.height * src.width;

	upload_image_buffers_to_device(bufs->src_bufs, src, has_gmask, stream);
	upload_image_buffers_to_device(bufs->tgt_bufs, tgt, has_gmask, stream);

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	nnf_randomize_kernel<<<blocks, num_threads, 0, stream>>>(
		d_field_ptr, bufs->src_bufs.rgb_mask, bufs->src_bufs.gx,
		bufs->src_bufs.gy, bufs->tgt_bufs.rgb_mask, bufs->tgt_bufs.gx,
		bufs->tgt_bufs.gy, has_gmask, src.height, src.width, tgt.height,
		tgt.width, patch_size, max_retry, reset, seed);
}

extern "C" void launch_nnf_initialize_from(
	CudaNNFDeviceBuffers *bufs, int *d_field_ptr, const int *other_d_field_ptr,
	const HostImageBuffers &src, const HostImageBuffers &tgt, int other_src_h,
	int other_src_w, bool has_gmask, int patch_size, int max_retry,
	unsigned int seed, cudaStream_t stream) {

	int src_size = src.height * src.width;

	upload_image_buffers_to_device(bufs->src_bufs, src, has_gmask, stream);
	upload_image_buffers_to_device(bufs->tgt_bufs, tgt, has_gmask, stream);

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	// bilinear lookup from previous level's field.
	nnf_initialize_from_kernel<<<blocks, num_threads, 0, stream>>>(
		d_field_ptr, other_d_field_ptr, bufs->src_bufs.rgb_mask,
		bufs->src_bufs.gx, bufs->src_bufs.gy, bufs->tgt_bufs.rgb_mask,
		bufs->tgt_bufs.gx, bufs->tgt_bufs.gy, has_gmask, src.height, src.width,
		tgt.height, tgt.width, other_src_h, other_src_w, patch_size);

	// randomize any entries whose distance is still >= kDistanceScale.
	nnf_randomize_kernel<<<blocks, num_threads, 0, stream>>>(
		d_field_ptr, bufs->src_bufs.rgb_mask, bufs->src_bufs.gx,
		bufs->src_bufs.gy, bufs->tgt_bufs.rgb_mask, bufs->tgt_bufs.gx,
		bufs->tgt_bufs.gy, has_gmask, src.height, src.width, tgt.height,
		tgt.width, patch_size, max_retry, false, seed ^ 0xDEADBEEFu);
}

__device__ bool d_is_patch_masked(const uchar4 *src_rgb_mask, bool has_gmask,
								  int y, int x, int h, int w, int patch_size) {

	for (int dy = -patch_size; dy <= patch_size; dy++) {
		for (int dx = -patch_size; dx <= patch_size; dx++) {
			int yy = y + dy, xx = x + dx;
			if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
				if ((src_rgb_mask[yy * w + xx].w & 1))
					return true;
				if (has_gmask && (src_rgb_mask[yy * w + xx].w & 2))
					return true;
			}
		}
	}
	return false;
}

__global__ void nnf_set_identity_kernel(int *field, const uchar4 *src_rgb_mask,
										bool has_gmask, int src_h, int src_w,
										int patch_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int y = idx / src_w, x = idx % src_w;

	if (has_gmask && (src_rgb_mask[idx].w & 2))
		return;

	if (!d_is_patch_masked(src_rgb_mask, has_gmask, y, x, src_h, src_w,
						   patch_size)) {
		int *nnf_field = field + idx * 3;
		nnf_field[0] = y;
		nnf_field[1] = x;
		nnf_field[2] = 0;
	}
}

extern "C" void launch_nnf_set_identity(CudaNNFDeviceBuffers *bufs,
										int *d_field_ptr,
										const HostImageBuffers &src,
										bool has_gmask, int patch_size) {

	int src_size = src.height * src.width;

	cudaCheckError(cudaMemcpy(bufs->src_bufs.rgb_mask, src.rgb_mask,
							  src_size * sizeof(uchar4),
							  cudaMemcpyHostToDevice));

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	nnf_set_identity_kernel<<<blocks, num_threads>>>(
		d_field_ptr, bufs->src_bufs.rgb_mask, has_gmask, src.height, src.width,
		patch_size);
}

__global__ void expectation_step_kernel(const uchar4 *iter_img,
										const uchar4 *peer_img, const int *nnf,
										float4 *vote, const float *dist2sim,
										bool source2target, bool has_gmask,
										int iter_h, int iter_w, int peer_h,
										int peer_w, int patch_size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= iter_h * iter_w)
		return;

	int i = idx / iter_w;
	int j = idx % iter_w;

	if (has_gmask && (iter_img[idx].w & 2))
		return;

	int yp = nnf[idx * 3 + 0];
	int xp = nnf[idx * 3 + 1];
	int dp = nnf[idx * 3 + 2];

	if (dp < 0)
		dp = 0;
	// dp is bounded by kDistanceScale=65535 in the kernels
	float w = dist2sim[dp];

	for (int di = -patch_size; di <= patch_size; ++di) {
		for (int dj = -patch_size; dj <= patch_size; ++dj) {
			int ya = i + di, xa = j + dj;	// iter side (NNF source domain)
			int yb = yp + di, xb = xp + dj; // peer side (NNF target domain)

			if (ya < 0 || ya >= iter_h || xa < 0 || xa >= iter_w)
				continue;
			if (yb < 0 || yb >= peer_h || xb < 0 || xb >= peer_w)
				continue;


			//   s2t (true):  read from iter (source), write to peer (target)
			//   t2s (false): read from peer (source), write to iter (target)
			uchar4 read_px;
			int vote_y, vote_x, vote_w_stride;
			if (source2target) {
				read_px = iter_img[ya * iter_w + xa]; // source color
				vote_y = yb;
				vote_x = xb;
				vote_w_stride = peer_w; // vote sized to target = peer
			} else {
				read_px = peer_img[yb * peer_w + xb]; // source color
				vote_y = ya;
				vote_x = xa;
				vote_w_stride = iter_w; // vote sized to target = iter
			}

			// skip if the source pixel we're voting from is masked
			if (read_px.w & 1)
				continue;
			if (has_gmask && (read_px.w & 2))
				continue;

			int vidx = vote_y * vote_w_stride + vote_x;

			atomicAdd(&vote[vidx].x, w * (float)read_px.x);
			atomicAdd(&vote[vidx].y, w * (float)read_px.y);
			atomicAdd(&vote[vidx].z, w * (float)read_px.z);
			atomicAdd(&vote[vidx].w, w);
		}
	}
}

__global__ void maximization_step_kernel(const float4 *vote,
										 uchar4 *tgt_rgb_mask, bool has_gmask,
										 int tgt_h, int tgt_w) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= tgt_h * tgt_w)
		return;

	uchar4 rm = tgt_rgb_mask[idx];
	if (has_gmask && (rm.w & 2))
		return; // gmask

	float4 v = vote[idx];
	if (v.w > 0) {
		rm.x = (unsigned char)fminf(255.f, fmaxf(0.f, v.x / v.w));
		rm.y = (unsigned char)fminf(255.f, fmaxf(0.f, v.y / v.w));
		rm.z = (unsigned char)fminf(255.f, fmaxf(0.f, v.z / v.w));
	} else {
		rm.w &= ~1;
	}
	tgt_rgb_mask[idx] = rm;
}

extern "C" void launch_em_iteration(CudaNNFDeviceBuffers *bufs, int src_h,
									int src_w, int tgt_h, int tgt_w,
									bool has_gmask, int patch_size,
									cudaStream_t stream) {

	int src_pixels = src_h * src_w;
	int tgt_pixels = tgt_h * tgt_w;

	int num_threads = 256;

	// Zero the vote buffer
	cudaCheckError(
		cudaMemsetAsync(bufs->vote, 0, tgt_pixels * sizeof(float4), stream));

	// E-step source -> target (iter over source domain)
	{
		int blocks = (src_pixels + num_threads - 1) / num_threads;
		expectation_step_kernel<<<blocks, num_threads, 0, stream>>>(
			bufs->src_bufs.rgb_mask, bufs->tgt_bufs.rgb_mask, bufs->s2t_curr,
			bufs->vote, bufs->dist2sim, true, has_gmask, src_h, src_w, tgt_h,
			tgt_w, patch_size);
	}

	// E-step target -> source (iter over target domain)
	{
		int blocks = (tgt_pixels + num_threads - 1) / num_threads;
		expectation_step_kernel<<<blocks, num_threads, 0, stream>>>(
			bufs->tgt_bufs.rgb_mask, bufs->src_bufs.rgb_mask, bufs->t2s_curr,
			bufs->vote, bufs->dist2sim, false, has_gmask, tgt_h, tgt_w, src_h,
			src_w, patch_size);
	}

	// M-step
	{
		int blocks = (tgt_pixels + num_threads - 1) / num_threads;
		maximization_step_kernel<<<blocks, num_threads, 0, stream>>>(
			bufs->vote, bufs->tgt_bufs.rgb_mask, has_gmask, tgt_h, tgt_w);
	}
	LOG("EM GPU done\n");
}