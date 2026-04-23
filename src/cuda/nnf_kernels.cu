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
	// int new_capacity_3c = num_pixels * 3;

	// if (num_pixels > pixel_capacity) {
	// 	realloc_device_ptr(&img, new_capacity_3c);
	// 	realloc_device_ptr(&gx, new_capacity_3c);
	// 	realloc_device_ptr(&gy, new_capacity_3c);
	// 	realloc_device_ptr(&mask, num_pixels);
	// 	if (has_gmask) {
	// 		realloc_device_ptr(&gmask, num_pixels);
	// 	}
	// 	pixel_capacity = num_pixels;
	// }

	if (num_pixels > pixel_capacity) {
		realloc_device_ptr(&data, num_pixels);
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
}

DeviceImageBuffers::~DeviceImageBuffers() {
	// if (img) {
	// 	cudaCheckError(cudaFree(img));
	// 	cudaCheckError(cudaFree(gx));
	// 	cudaCheckError(cudaFree(gy));
	// 	cudaCheckError(cudaFree(mask));
	// }
	// if (gmask)
	// 	cudaCheckError(cudaFree(gmask));
	if (data)
		cudaCheckError(cudaFree(data));
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
}

__device__ int compute_patch_dist(const PixelData *src_data,
								  const PixelData *tgt_data, bool has_gmask,
								  int ys, int xs, int yt, int xt, int src_h,
								  int src_w, int tgt_h, int tgt_w,
								  int patch_size) {
	float distance = 0;
	float wsum = 0;
	const float kSSDScale = 9 * 255 * 255;
	const float kDistanceScale = 65535;

	for (int dy = -patch_size; dy <= patch_size; ++dy) {
		const int yys = ys + dy, yyt = yt + dy;

		if (yys <= 0 || yys >= src_h - 1 || yyt <= 0 || yyt >= tgt_h - 1) {
			distance += kSSDScale * (2 * patch_size + 1);
			wsum += 2 * patch_size + 1;
			continue;
		}

		for (int dx = -patch_size; dx <= patch_size; ++dx) {
			int xxs = xs + dx, xxt = xt + dx;
			wsum += 1;

			if (xxs <= 0 || xxs >= src_w - 1 || xxt <= 0 || xxt >= tgt_w - 1) {
				distance += kSSDScale;
				continue;
			}

			const PixelData &src_p = src_data[yys * src_w + xxs];
			const PixelData &tgt_p = tgt_data[yyt * tgt_w + xxt];

			bool is_masked = src_p.mask || tgt_p.mask ||
							 (has_gmask && src_p.gmask) ||
							 (has_gmask && tgt_p.gmask);
			if (is_masked) {
				distance += kSSDScale;
				continue;
			}

			int ssd = 0;
			// for (int c = 0; c < 3; c++) {
			// 	int s_value = src_img[(yys * src_w + xxs) * 3 + c];
			// 	int t_value = tgt_img[(yyt * tgt_w + xxt) * 3 + c];
			// 	int s_gy = src_gy[(yys * src_w + xxs) * 3 + c];
			// 	int s_gx = src_gx[(yys * src_w + xxs) * 3 + c];
			// 	int t_gy = tgt_gy[(yyt * tgt_w + xxt) * 3 + c];
			// 	int t_gx = tgt_gx[(yyt * tgt_w + xxt) * 3 + c];

			// 	ssd += (s_value - t_value) * (s_value - t_value);
			// 	ssd += (s_gx - t_gx) * (s_gx - t_gx);
			// 	ssd += (s_gy - t_gy) * (s_gy - t_gy);
			// }
			// rgb channels
			ssd += (src_p.rgb.x - tgt_p.rgb.x) * (src_p.rgb.x - tgt_p.rgb.x);
			ssd += (src_p.rgb.y - tgt_p.rgb.y) * (src_p.rgb.y - tgt_p.rgb.y);
			ssd += (src_p.rgb.z - tgt_p.rgb.z) * (src_p.rgb.z - tgt_p.rgb.z);
			// gradx
			ssd += (src_p.gx.x - tgt_p.gx.x) * (src_p.gx.x - tgt_p.gx.x);
			ssd += (src_p.gx.y - tgt_p.gx.y) * (src_p.gx.y - tgt_p.gx.y);
			ssd += (src_p.gx.z - tgt_p.gx.z) * (src_p.gx.z - tgt_p.gx.z);
			// grad y
			ssd += (src_p.gy.x - tgt_p.gy.x) * (src_p.gy.x - tgt_p.gy.x);
			ssd += (src_p.gy.y - tgt_p.gy.y) * (src_p.gy.y - tgt_p.gy.y);
			ssd += (src_p.gy.z - tgt_p.gy.z) * (src_p.gy.z - tgt_p.gy.z);
			distance += ssd;
		}
	}

	distance /= kSSDScale;

	int res = (int)(kDistanceScale * distance / wsum);
	if (res < 0 || res > kDistanceScale)
		return kDistanceScale;
	return res;
}

/**
 * Copies the image host buffers to the device
 */
// static void upload_image_buffers_to_device(DeviceImageBuffers &d_bufs,
// 										   const HostImageBuffers &h_bufs,
// 										   bool has_gmask) {
// 	int size = h_bufs.height * h_bufs.width;
// 	cudaCheckError(
// 		cudaMemcpy(d_bufs.img, h_bufs.img, size * 3, cudaMemcpyHostToDevice));
// 	cudaCheckError(
// 		cudaMemcpy(d_bufs.gx, h_bufs.gx, size * 3, cudaMemcpyHostToDevice));
// 	cudaCheckError(
// 		cudaMemcpy(d_bufs.gy, h_bufs.gy, size * 3, cudaMemcpyHostToDevice));
// 	cudaCheckError(
// 		cudaMemcpy(d_bufs.mask, h_bufs.mask, size, cudaMemcpyHostToDevice));
// 	if (has_gmask) {
// 		cudaCheckError(cudaMemcpy(d_bufs.gmask, h_bufs.gmask, size,
// 								  cudaMemcpyHostToDevice));
// 	}
// }

__global__ void nnf_jump_flood_kernel(const int *field_in, int *field_out,
									  const PixelData *src_data,
									  const PixelData *tgt_data, bool has_gmask,
									  int src_h, int src_w, int tgt_h,
									  int tgt_w, int patch_size,
									  unsigned int seed, int jump) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int p_y = idx / src_w, p_x = idx % src_w;

	// Skip pixels that are globally masked in the source -- nothing to match.
	if (has_gmask && src_data[idx].gmask)
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
		if (has_gmask && src_data[newidx].gmask)
			continue;

		const int *neighbor_field = field_in + (newy * src_w + newx) * 3;
		int candidate_y =
			device_clamp(neighbor_field[0] - directions[i][0], 0, tgt_h - 1);
		int candidate_x =
			device_clamp(neighbor_field[1] - directions[i][1], 0, tgt_w - 1);

		int computed_dist = compute_patch_dist(
			src_data, tgt_data, has_gmask, p_y, p_x, candidate_y, candidate_x,
			src_h, src_w, tgt_h, tgt_w, patch_size);
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

		if (!(has_gmask && tgt_data[idxp].gmask)) {
			int d =
				compute_patch_dist(src_data, tgt_data, has_gmask, p_y, p_x, yp,
								   xp, src_h, src_w, tgt_h, tgt_w, patch_size);
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

// __global__ void nnf_minimize_kernel(
// 	int *field, const unsigned char *src_img, const unsigned char *tgt_img,
// 	const unsigned char *src_gx, const unsigned char *src_gy,
// 	const unsigned char *tgt_gx, const unsigned char *tgt_gy,
// 	const unsigned char *src_mask, const unsigned char *tgt_mask,
// 	const unsigned char *src_gmask, const unsigned char *tgt_gmask,
// 	bool has_gmask, int src_h, int src_w, int tgt_h, int tgt_w, int patch_size,
// 	int color, unsigned int seed) {
// 	int idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (idx >= src_h * src_w)
// 		return;

// 	int p_y = idx / src_w, p_x = idx % src_w; // pixel coordinates

// 	// even sum coords - red
// 	// odd sum coords - black
// 	if ((p_x + p_y) % 2 != color)
// 		return;

// 	int *nnf_field = field + idx * 3; // 3 channels
// 	int nnf_best_y = nnf_field[0];
// 	int nnf_best_x = nnf_field[1];
// 	int nnf_best_d = nnf_field[2];

// 	const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

// 	// propagation phase
// 	for (int i = 0; i < 4; i++) {
// 		int newy = p_y + directions[i][0];
// 		int newx = p_x + directions[i][1];

// 		if (newy < 0 || newy >= src_h || newx < 0 || newx >= src_w)
// 			continue;
// 		if (has_gmask && src_gmask[newy * src_w + newx])
// 			continue;

// 		const int *neighbor_field = field + (newy * src_w + newx) * 3;
// 		int clamped_y =
// 			device_clamp(neighbor_field[0] - directions[i][0], 0, tgt_h - 1);
// 		int clamped_x =
// 			device_clamp(neighbor_field[1] - directions[i][1], 0, tgt_w - 1);

// 		int computed_dist = compute_patch_dist(
// 			src_img, tgt_img, src_gx, src_gy, tgt_gx, tgt_gy, src_mask,
// 			tgt_mask, src_gmask, tgt_gmask, has_gmask, p_y, p_x, clamped_y,
// 			clamped_x, src_h, src_w, tgt_h, tgt_w, patch_size);
// 		if (computed_dist < nnf_best_d) {
// 			nnf_best_x = newx;
// 			nnf_best_y = newy;
// 			nnf_best_d = computed_dist;
// 		}
// 	}

// 	// random search phase
// 	int random_scale = (min(tgt_h, tgt_w) - 1) / 2;
// 	int step = 0;

// 	while (random_scale > 0) {
// 		// coprimes used to reduce collisions
// 		int yp = device_clamp(nnf_best_y +
// 								  rand_range(seed + idx * 1337u + step * 7919u,
// 											 -random_scale, random_scale),
// 							  0, tgt_h - 1);
// 		int xp = device_clamp(nnf_best_x +
// 								  rand_range(seed + idx * 7919u + step * 1337u,
// 											 -random_scale, random_scale),
// 							  0, tgt_w - 1);

// 		if (has_gmask && tgt_gmask[yp * tgt_w + xp]) {
// 			random_scale /= 2;
// 		}

// 		int dp = compute_patch_dist(src_img, tgt_img, src_gx, src_gy, tgt_gx,
// 									tgt_gy, src_mask, tgt_mask, src_gmask,
// 									tgt_gmask, has_gmask, p_y, p_x, yp, xp,
// 									src_h, src_w, tgt_h, tgt_w, patch_size);
// 		if (dp < nnf_best_d) {
// 			nnf_best_x = xp;
// 			nnf_best_y = yp;
// 			nnf_best_d = dp;
// 		}
// 		random_scale /= 2;
// 		step++;
// 	}

// 	nnf_field[0] = nnf_best_y;
// 	nnf_field[1] = nnf_best_x;
// 	nnf_field[2] = nnf_best_d;
// }

extern "C" void launch_nnf_minimize(
	CudaNNFDeviceBuffers *bufs, int *d_field_ptr, int *d_field_scratch,
	int *h_field_ptr, const HostImageBuffers &src, const HostImageBuffers &tgt,
	bool has_gmask, int patch_size, int nr_pass, unsigned int random_seed) {

	int src_size = src.height * src.width;
	int tgt_size = tgt.height * tgt.width;

	// Timing setup
	cudaEvent_t t0, t1, t2, t3;
	cudaEventCreate(&t0);
	cudaEventCreate(&t1);
	cudaEventCreate(&t2);
	cudaEventCreate(&t3);

	cudaEventRecord(t0);

	// upload_image_buffers_to_device(bufs->src_bufs, src, has_gmask);
	// upload_image_buffers_to_device(bufs->tgt_bufs, tgt, has_gmask);

	cudaCheckError(cudaMemcpy(bufs->src_bufs.data, src.data,
							  src_size * sizeof(PixelData),
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->tgt_bufs.data, tgt.data,
							  tgt_size * sizeof(PixelData),
							  cudaMemcpyHostToDevice));

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
				in_ptr, out_ptr, bufs->src_bufs.data, bufs->tgt_bufs.data,
				has_gmask, src.height, src.width, tgt.height, tgt.width,
				patch_size, seed, jumps[ji]);

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
__global__ void nnf_randomize_kernel(int *field, const PixelData *src_data,
									 const PixelData *tgt_data, bool has_gmask,
									 int src_h, int src_w, int tgt_h, int tgt_w,
									 int patch_size, int max_retry, bool reset,
									 unsigned int seed) {

	const int kDistanceScale = 65535;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int p_y = idx / src_w, p_x = idx % src_w; // pixel coordinates

	if (has_gmask && src_data[idx].gmask)
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

		if (has_gmask && tgt_data[y_t * tgt_w + x_t].gmask)
			continue;

		int d = compute_patch_dist(src_data, tgt_data, has_gmask, p_y, p_x, y_t,
								   x_t, src_h, src_w, tgt_h, tgt_w, patch_size);

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
	int *field, const int *other_field, const PixelData *src_data,
	const PixelData *tgt_data, bool has_gmask, int src_h, int src_w, int tgt_h,
	int tgt_w, int other_src_h, int other_src_w, int patch_size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int i = idx / src_w, j = idx % src_w; // pixel coordinates

	if (has_gmask && src_data[idx].gmask)
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

	int dist = compute_patch_dist(src_data, tgt_data, has_gmask, i, j, y_t, x_t,
								  src_h, src_w, tgt_h, tgt_w, patch_size);

	int *nnf_field = field + idx * 3;
	nnf_field[0] = y_t;
	nnf_field[1] = x_t;
	nnf_field[2] = dist;
}

extern "C" void
launch_nnf_randomize(CudaNNFDeviceBuffers *bufs, int *d_field_ptr,
					 const HostImageBuffers &src, const HostImageBuffers &tgt,
					 bool has_gmask, int patch_size, int max_retry, bool reset,
					 unsigned int seed) {

	int src_size = src.height * src.width;
	int tgt_size = tgt.height * tgt.width;

	// upload_image_buffers_to_device(bufs->src_bufs, src, has_gmask);
	// upload_image_buffers_to_device(bufs->tgt_bufs, tgt, has_gmask);
	cudaCheckError(cudaMemcpy(bufs->src_bufs.data, src.data,
							  src_size * sizeof(PixelData),
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->tgt_bufs.data, tgt.data,
							  tgt_size * sizeof(PixelData),
							  cudaMemcpyHostToDevice));

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	nnf_randomize_kernel<<<blocks, num_threads>>>(
		d_field_ptr, bufs->src_bufs.data, bufs->tgt_bufs.data, has_gmask,
		src.height, src.width, tgt.height, tgt.width, patch_size, max_retry,
		reset, seed);
}

extern "C" void launch_nnf_initialize_from(
	CudaNNFDeviceBuffers *bufs, int *d_field_ptr, const int *other_d_field_ptr,
	const HostImageBuffers &src, const HostImageBuffers &tgt, int other_src_h,
	int other_src_w, bool has_gmask, int patch_size, int max_retry,
	unsigned int seed) {

	int src_size = src.height * src.width;
	int tgt_size = tgt.height * tgt.width;

	// upload_image_buffers_to_device(bufs->src_bufs, src, has_gmask);
	// upload_image_buffers_to_device(bufs->tgt_bufs, tgt, has_gmask);
	cudaCheckError(cudaMemcpy(bufs->src_bufs.data, src.data,
							  src_size * sizeof(PixelData),
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->tgt_bufs.data, tgt.data,
							  tgt_size * sizeof(PixelData),
							  cudaMemcpyHostToDevice));

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	// bilinear lookup from previous level's field.
	nnf_initialize_from_kernel<<<blocks, num_threads>>>(
		d_field_ptr, other_d_field_ptr, bufs->src_bufs.data,
		bufs->tgt_bufs.data, has_gmask, src.height, src.width, tgt.height,
		tgt.width, other_src_h, other_src_w, patch_size);

	// randomize any entries whose distance is still >= kDistanceScale.
	nnf_randomize_kernel<<<blocks, num_threads>>>(
		d_field_ptr, bufs->src_bufs.data, bufs->tgt_bufs.data, has_gmask,
		src.height, src.width, tgt.height, tgt.width, patch_size, max_retry,
		false, seed ^ 0xDEADBEEFu);
}

__device__ bool d_is_patch_masked(const PixelData *data, bool has_gmask, int y,
								  int x, int h, int w, int patch_size) {

	for (int dy = -patch_size; dy <= patch_size; dy++) {
		for (int dx = -patch_size; dx <= patch_size; dx++) {
			int yy = y + dy, xx = x + dx;
			if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
				if (data[yy * w + xx].mask)
					return true;
				if (has_gmask && data[yy * w + xx].gmask)
					return true;
			}
		}
	}
	return false;
}

__global__ void nnf_set_identity_kernel(int *field, const PixelData *data,
										bool has_gmask, int src_h, int src_w,
										int patch_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int y = idx / src_w, x = idx % src_w;

	if (has_gmask && data[idx].gmask)
		return;

	if (!d_is_patch_masked(data, has_gmask, y, x, src_h, src_w, patch_size)) {
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

	cudaCheckError(cudaMemcpy(bufs->src_bufs.data, src.data,
							  src_size * sizeof(PixelData),
							  cudaMemcpyHostToDevice));

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	nnf_set_identity_kernel<<<blocks, num_threads>>>(
		d_field_ptr, bufs->src_bufs.data, has_gmask, src.height, src.width,
		patch_size);
}