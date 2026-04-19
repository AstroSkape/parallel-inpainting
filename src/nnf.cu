#include "../include/cuda_helpers.cuh"
#include "../include/cuda_helpers.h"
#include <cuda_runtime.h>
#include <iostream>

#define RED 0
#define BLACK 1

#define DIR_S2T 0 // source -> target: src=A, tgt=B, writes field_s2t
#define DIR_T2S 1 // target -> source: src=B, tgt=A, writes field_t2s

// Wang hash
// Code adapted from https://burtleburtle.net/bob/hash/integer.html
__device__ unsigned int wang_hash(unsigned int a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
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

void realloc_device_ptr(unsigned char **ptr, int new_bytes) {
	if (*ptr)
		cudaCheckError(cudaFree(*ptr));
	cudaCheckError(cudaMalloc(ptr, new_bytes));
}

/**
 * Allocates space only if needed. Max sized buffers are retained
 */
void DeviceImageBuffers::allocate_buffers(int num_pixels, bool has_gmask) {
	int prev_capacity_3c = pixel_capacity * 3;
	int new_capacity_3c = num_pixels * 3;

	if (new_capacity_3c > prev_capacity_3c) {
		realloc_device_ptr(&img, new_capacity_3c);
		realloc_device_ptr(&gx, new_capacity_3c);
		realloc_device_ptr(&gy, new_capacity_3c);
		realloc_device_ptr(&mask, num_pixels);
		if (has_gmask) {
			realloc_device_ptr(&gmask, num_pixels);
		}
	}

	if (num_pixels > pixel_capacity)
		pixel_capacity = num_pixels;
}

/**
 * Ensures that the buffers can hold the required number of pixels.
 * Reuses existing buffers to avoid repeated calls to malloc/free
 */
void CudaNNFDeviceBuffers::allocate_device_buffers(int src_pixels,
												   int tgt_pixels,
												   bool need_gmask) {
	if (src_pixels > src_bufs.pixel_capacity) {
		if (field_ptr)
			cudaCheckError(cudaFree(field_ptr));
		cudaCheckError(cudaMalloc(&field_ptr, src_pixels * 3 * sizeof(int)));
	}

	src_bufs.allocate_buffers(src_pixels, need_gmask);
	tgt_bufs.allocate_buffers(tgt_pixels, need_gmask);
}

void CudaFusedNNFDeviceBuffers::allocate_device_buffers(int a_pixels,
														int b_pixels,
														bool need_gmask) {
	// s2t field: sized to img_a (s2t's source).
	if (a_pixels > field_capacity_s2t) {
		if (field_ptr_s2t)
			cudaCheckError(cudaFree(field_ptr_s2t));
		cudaCheckError(cudaMalloc(&field_ptr_s2t, a_pixels * 3 * sizeof(int)));
		field_capacity_s2t = a_pixels;
	}

	// t2s field: sized to img_b (t2s's source).
	if (b_pixels > field_capacity_t2s) {
		if (field_ptr_t2s)
			cudaCheckError(cudaFree(field_ptr_t2s));
		cudaCheckError(cudaMalloc(&field_ptr_t2s, b_pixels * 3 * sizeof(int)));
		field_capacity_t2s = b_pixels;
	}

	a_bufs.allocate_buffers(a_pixels, need_gmask);
	b_bufs.allocate_buffers(b_pixels, need_gmask);
}

DeviceImageBuffers::~DeviceImageBuffers() {
	if (img) {
		cudaCheckError(cudaFree(img));
		cudaCheckError(cudaFree(gx));
		cudaCheckError(cudaFree(gy));
		cudaCheckError(cudaFree(mask));
	}
	if (gmask)
		cudaCheckError(cudaFree(gmask));
}

CudaFusedNNFDeviceBuffers::~CudaFusedNNFDeviceBuffers() {
	if (field_ptr_s2t)
		cudaCheckError(cudaFree(field_ptr_s2t));
	if (field_ptr_t2s)
		cudaCheckError(cudaFree(field_ptr_t2s));
}

CudaNNFDeviceBuffers::~CudaNNFDeviceBuffers() {
	if (field_ptr) {
		cudaCheckError(cudaFree(field_ptr));
	}
}

__device__ int
compute_patch_dist(const unsigned char *src_img, const unsigned char *tgt_img,
				   const unsigned char *src_gx, const unsigned char *src_gy,
				   const unsigned char *tgt_gx, const unsigned char *tgt_gy,
				   const unsigned char *src_mask, const unsigned char *tgt_mask,
				   const unsigned char *src_gmask,
				   const unsigned char *tgt_gmask, bool has_gmask, int ys,
				   int xs, int yt, int xt, int src_h, int src_w, int tgt_h,
				   int tgt_w, int patch_size) {
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

			bool is_masked = src_mask[yys * src_w + xxs] ||
							 tgt_mask[yyt * tgt_w + xxt] ||
							 (has_gmask && src_gmask[yys * src_w + xxs]) ||
							 (has_gmask && tgt_gmask[yyt * tgt_w + xxt]);
			if (is_masked) {
				distance += kSSDScale;
				continue;
			}

			int ssd = 0;
			for (int c = 0; c < 3; c++) {
				int s_value = src_img[(yys * src_w + xxs) * 3 + c];
				int t_value = tgt_img[(yyt * tgt_w + xxt) * 3 + c];
				int s_gy = src_gy[(yys * src_w + xxs) * 3 + c];
				int s_gx = src_gx[(yys * src_w + xxs) * 3 + c];
				int t_gy = tgt_gy[(yyt * tgt_w + xxt) * 3 + c];
				int t_gx = tgt_gx[(yyt * tgt_w + xxt) * 3 + c];

				ssd += (s_value - t_value) * (s_value - t_value);
				ssd += (s_gx - t_gx) * (s_gx - t_gx);
				ssd += (s_gy - t_gy) * (s_gy - t_gy);
			}
			distance += ssd;
		}
	}

	distance /= kSSDScale;

	int res = (int)(kDistanceScale * distance / wsum);
	if (res < 0 || res > kDistanceScale)
		return kDistanceScale;
	return res;
}

__global__ void nnf_minimize_kernel(
	int *field, const unsigned char *src_img, const unsigned char *tgt_img,
	const unsigned char *src_gx, const unsigned char *src_gy,
	const unsigned char *tgt_gx, const unsigned char *tgt_gy,
	const unsigned char *src_mask, const unsigned char *tgt_mask,
	const unsigned char *src_gmask, const unsigned char *tgt_gmask,
	bool has_gmask, int src_h, int src_w, int tgt_h, int tgt_w, int patch_size,
	int color, unsigned int seed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= src_h * src_w)
		return;

	int p_y = idx / src_w, p_x = idx % src_w; // pixel coordinates

	// even sum coords - red
	// odd sum coords - black
	if ((p_x + p_y) % 2 != color)
		return;

	int *nnf_field = field + idx * 3; // 3 channels
	int nnf_best_y = nnf_field[0];
	int nnf_best_x = nnf_field[1];
	int nnf_best_d = nnf_field[2];

	const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	// propagation phase
	for (int i = 0; i < 4; i++) {
		int newy = p_y + directions[i][0];
		int newx = p_x + directions[i][1];

		if (newy < 0 || newy >= src_h || newx < 0 || newx >= src_w)
			continue;
		if (has_gmask && src_gmask[newy * src_w + newx])
			continue;

		const int *neighbor_field = field + (newy * src_w + newx) * 3;
		int clamped_y =
			device_clamp(neighbor_field[0] - directions[i][0], 0, tgt_h - 1);
		int clamped_x =
			device_clamp(neighbor_field[1] - directions[i][1], 0, tgt_w - 1);

		int computed_dist = compute_patch_dist(
			src_img, tgt_img, src_gx, src_gy, tgt_gx, tgt_gy, src_mask,
			tgt_mask, src_gmask, tgt_gmask, has_gmask, p_y, p_x, clamped_y,
			clamped_x, src_h, src_w, tgt_h, tgt_w, patch_size);
		if (computed_dist < nnf_best_d) {
			nnf_best_x = newx;
			nnf_best_y = newy;
			nnf_best_d = computed_dist;
		}
	}

	// random search phase
	int random_scale = (min(tgt_h, tgt_w) - 1) / 2;
	int step = 0;

	while (random_scale > 0) {
		// coprimes used to reduce collisions
		int yp = device_clamp(nnf_best_y +
								  rand_range(seed + idx * 1337u + step * 7919u,
											 -random_scale, random_scale),
							  0, tgt_h - 1);
		int xp = device_clamp(nnf_best_x +
								  rand_range(seed + idx * 1337u + step * 7919u,
											 -random_scale, random_scale),
							  0, tgt_w - 1);

		if (has_gmask && tgt_gmask[yp * tgt_w + xp]) {
			random_scale /= 2;
		}

		int dp = compute_patch_dist(src_img, tgt_img, src_gx, src_gy, tgt_gx,
									tgt_gy, src_mask, tgt_mask, src_gmask,
									tgt_gmask, has_gmask, p_y, p_x, yp, xp,
									src_h, src_w, tgt_h, tgt_w, patch_size);
		if (dp < nnf_best_d) {
			nnf_best_x = xp;
			nnf_best_y = yp;
			nnf_best_d = dp;
		}
		random_scale /= 2;
		step++;
	}

	nnf_field[0] = nnf_best_y;
	nnf_field[1] = nnf_best_x;
	nnf_field[2] = nnf_best_d;
}

/**
 * Fused Kernel
 *
 * Grid Layout:
 * 	gridDim.x = ceil(max(a_pixels, b_pixels) / blockDim.x)
 * 	gridDim.y = 2 (0 = s2t, 1 = t2s)
 */
__global__ void nnf_minimize_fused_kernel(
	int *field_s2t, int *field_t2s,
	// Image A (= s2t's source, = t2s's target)
	const unsigned char *a_img, const unsigned char *a_gx,
	const unsigned char *a_gy, const unsigned char *a_mask,
	const unsigned char *a_gmask,
	// Image B (= s2t's target, = t2s's source)
	const unsigned char *b_img, const unsigned char *b_gx,
	const unsigned char *b_gy, const unsigned char *b_mask,
	const unsigned char *b_gmask, bool has_gmask, int a_h, int a_w, int b_h,
	int b_w, int patch_size, int color, unsigned int seed) {

	const int dir = blockIdx.y; // 0 or 1

	int *field;
	const unsigned char *src_img, *src_gx, *src_gy, *src_mask, *src_gmask;
	const unsigned char *tgt_img, *tgt_gx, *tgt_gy, *tgt_mask, *tgt_gmask;
	int src_h, src_w, tgt_h, tgt_w;

	if (dir == DIR_S2T) {
		field = field_s2t;
		src_img = a_img;
		src_gx = a_gx;
		src_gy = a_gy;
		src_mask = a_mask;
		src_gmask = a_gmask;
		tgt_img = b_img;
		tgt_gx = b_gx;
		tgt_gy = b_gy;
		tgt_mask = b_mask;
		tgt_gmask = b_gmask;
		src_h = a_h;
		src_w = a_w;
		tgt_h = b_h;
		tgt_w = b_w;
	} else {
		field = field_t2s;
		src_img = b_img;
		src_gx = b_gx;
		src_gy = b_gy;
		src_mask = b_mask;
		src_gmask = b_gmask;
		tgt_img = a_img;
		tgt_gx = a_gx;
		tgt_gy = a_gy;
		tgt_mask = a_mask;
		tgt_gmask = a_gmask;
		src_h = b_h;
		src_w = b_w;
		tgt_h = a_h;
		tgt_w = a_w;
	}

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int src_size = src_h * src_w;
	if (idx >= src_size)
		return;

	const int p_y = idx / src_w;
	const int p_x = idx % src_w;

	// even sum coords - red
	// odd sum coords - black
	if ((p_x + p_y) % 2 != color)
		return;

	int *nnf_field = field + idx * 3;
	int nnf_best_y = nnf_field[0];
	int nnf_best_x = nnf_field[1];
	int nnf_best_d = nnf_field[2];

	const int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	// propagation phase
	for (int i = 0; i < 4; i++) {
		int newy = p_y + directions[i][0];
		int newx = p_x + directions[i][1];

		if (newy < 0 || newy >= src_h || newx < 0 || newx >= src_w)
			continue;
		if (has_gmask && src_gmask[newy * src_w + newx])
			continue;

		const int *neighbor_field = field + (newy * src_w + newx) * 3;
		int clamped_y =
			device_clamp(neighbor_field[0] - directions[i][0], 0, tgt_h - 1);
		int clamped_x =
			device_clamp(neighbor_field[1] - directions[i][1], 0, tgt_w - 1);

		int computed_dist = compute_patch_dist(
			src_img, tgt_img, src_gx, src_gy, tgt_gx, tgt_gy, src_mask,
			tgt_mask, src_gmask, tgt_gmask, has_gmask, p_y, p_x, clamped_y,
			clamped_x, src_h, src_w, tgt_h, tgt_w, patch_size);
		if (computed_dist < nnf_best_d) {
			nnf_best_x = newx;
			nnf_best_y = newy;
			nnf_best_d = computed_dist;
		}
	}

	// random search phase
	// a different seed depending on direction
	const unsigned int dir_seed = seed + (dir == DIR_S2T ? 0u : 0x9E3779B9u);

	int random_scale = (min(tgt_h, tgt_w) - 1) / 2;
	int step = 0;

	while (random_scale > 0) {
		int yp = device_clamp(
			nnf_best_y + rand_range(dir_seed + idx * 1337u + step * 7919u,
									-random_scale, random_scale),
			0, tgt_h - 1);
		int xp = device_clamp(
			nnf_best_x + rand_range(dir_seed + idx * 1337u + step * 7919u,
									-random_scale, random_scale),
			0, tgt_w - 1);

		if (has_gmask && tgt_gmask[yp * tgt_w + xp]) {
			random_scale /= 2;
		}

		int dp = compute_patch_dist(src_img, tgt_img, src_gx, src_gy, tgt_gx,
									tgt_gy, src_mask, tgt_mask, src_gmask,
									tgt_gmask, has_gmask, p_y, p_x, yp, xp,
									src_h, src_w, tgt_h, tgt_w, patch_size);
		if (dp < nnf_best_d) {
			nnf_best_x = xp;
			nnf_best_y = yp;
			nnf_best_d = dp;
		}
		random_scale /= 2;
		step++;
	}

	nnf_field[0] = nnf_best_y;
    nnf_field[1] = nnf_best_x;
    nnf_field[2] = nnf_best_d;
}

extern "C" void launch_fused_nnf_minimize(
	CudaFusedNNFDeviceBuffers *bufs, int *field_s2t_host, int *field_t2s_host,
	const HostImageBuffers &a, // s2t's source, t2s's target
	const HostImageBuffers &b, // s2t's target, t2s's source
	bool has_gmask, int patch_size, int nr_pass, unsigned int random_seed) {
	const int a_size = a.height * a.width;
	const int b_size = b.height * b.width;

	// copy from host to device
	cudaCheckError(cudaMemcpy(bufs->field_ptr_s2t, field_s2t_host,
							  a_size * 3 * sizeof(int),
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->field_ptr_t2s, field_t2s_host,
							  b_size * 3 * sizeof(int),
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->a_bufs.img, a.img, a_size * 3,
							  cudaMemcpyHostToDevice));
	cudaCheckError(
		cudaMemcpy(bufs->a_bufs.gx, a.gx, a_size * 3, cudaMemcpyHostToDevice));
	cudaCheckError(
		cudaMemcpy(bufs->a_bufs.gy, a.gy, a_size * 3, cudaMemcpyHostToDevice));
	cudaCheckError(
		cudaMemcpy(bufs->a_bufs.mask, a.mask, a_size, cudaMemcpyHostToDevice));

	cudaCheckError(cudaMemcpy(bufs->b_bufs.img, b.img, b_size * 3,
							  cudaMemcpyHostToDevice));
	cudaCheckError(
		cudaMemcpy(bufs->b_bufs.gx, b.gx, b_size * 3, cudaMemcpyHostToDevice));
	cudaCheckError(
		cudaMemcpy(bufs->b_bufs.gy, b.gy, b_size * 3, cudaMemcpyHostToDevice));
	cudaCheckError(
		cudaMemcpy(bufs->b_bufs.mask, b.mask, b_size, cudaMemcpyHostToDevice));
	if (has_gmask) {
		cudaCheckError(cudaMemcpy(bufs->a_bufs.gmask, a.gmask, a_size,
								  cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemcpy(bufs->b_bufs.gmask, b.gmask, b_size,
								  cudaMemcpyHostToDevice));
	}

	const int num_threads = 256;
	const int max_size = (a_size > b_size) ? a_size : b_size;
	const int blocks_x = (max_size + num_threads - 1) / num_threads;

	dim3 grid(blocks_x, 2);
	dim3 block(num_threads);

	LOG("[CUDA-FUSED] grid=(%d,2) block=%d\n", blocks_x, num_threads);

	for (int i = 0; i < nr_pass; i++) {
		unsigned int seed = random_seed + i * 12345u;

		// Red phase: both directions together.
		nnf_minimize_fused_kernel<<<grid, block>>>(
			bufs->field_ptr_s2t, bufs->field_ptr_t2s, bufs->a_bufs.img,
			bufs->a_bufs.gx, bufs->a_bufs.gy, bufs->a_bufs.mask,
			bufs->a_bufs.gmask, bufs->b_bufs.img, bufs->b_bufs.gx,
			bufs->b_bufs.gy, bufs->b_bufs.mask, bufs->b_bufs.gmask, has_gmask,
			a.height, a.width, b.height, b.width, patch_size, RED, seed);

		// Black phase: both directions together.
		nnf_minimize_fused_kernel<<<grid, block>>>(
			bufs->field_ptr_s2t, bufs->field_ptr_t2s, bufs->a_bufs.img,
			bufs->a_bufs.gx, bufs->a_bufs.gy, bufs->a_bufs.mask,
			bufs->a_bufs.gmask, bufs->b_bufs.img, bufs->b_bufs.gx,
			bufs->b_bufs.gy, bufs->b_bufs.mask, bufs->b_bufs.gmask, has_gmask,
			a.height, a.width, b.height, b.width, patch_size, BLACK, seed);
	}

	// --- D2H copies ---------------------------------------------------------
	cudaCheckError(cudaMemcpy(field_s2t_host, bufs->field_ptr_s2t,
							  a_size * 3 * sizeof(int),
							  cudaMemcpyDeviceToHost));
	cudaCheckError(cudaMemcpy(field_t2s_host, bufs->field_ptr_t2s,
							  b_size * 3 * sizeof(int),
							  cudaMemcpyDeviceToHost));
}

extern "C" void launch_nnf_minimize(CudaNNFDeviceBuffers *bufs, int *field_ptr,
									const HostImageBuffers &src,
									const HostImageBuffers &tgt, bool has_gmask,
									int patch_size, int nr_pass,
									unsigned int random_seed) {

	int src_size = src.height * src.width;
	int tgt_size = tgt.height * tgt.width;

	// copy from host to device
	cudaCheckError(cudaMemcpy(bufs->field_ptr, field_ptr,
							  src_size * 3 * sizeof(int),
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->src_bufs.img, src.img, src_size * 3,
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->tgt_bufs.img, tgt.img, tgt_size * 3,
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->src_bufs.gx, src.gx, src_size * 3,
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->src_bufs.gy, src.gy, src_size * 3,
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->tgt_bufs.gx, tgt.gx, tgt_size * 3,
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->tgt_bufs.gy, tgt.gy, tgt_size * 3,
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->src_bufs.mask, src.mask, src_size,
							  cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(bufs->tgt_bufs.mask, tgt.mask, tgt_size,
							  cudaMemcpyHostToDevice));
	if (has_gmask) {
		cudaCheckError(cudaMemcpy(bufs->src_bufs.gmask, src.gmask, src_size,
								  cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemcpy(bufs->tgt_bufs.gmask, tgt.gmask, tgt_size,
								  cudaMemcpyHostToDevice));
	}

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	LOG("[CUDA] Number of blocks: %d\n", blocks);

	for (int i = 0; i < nr_pass; i++) {
		unsigned int seed = random_seed + i * 12345u;

		// call kernel on red pixels
		nnf_minimize_kernel<<<blocks, num_threads>>>(
			bufs->field_ptr, bufs->src_bufs.img, bufs->tgt_bufs.img,
			bufs->src_bufs.gx, bufs->src_bufs.gy, bufs->tgt_bufs.gx,
			bufs->tgt_bufs.gy, bufs->src_bufs.mask, bufs->tgt_bufs.mask,
			bufs->src_bufs.gmask, bufs->tgt_bufs.gmask, has_gmask, src.height,
			src.width, tgt.height, tgt.width, patch_size, RED, seed);
		// call kernel on black pixels
		nnf_minimize_kernel<<<blocks, num_threads>>>(
			bufs->field_ptr, bufs->src_bufs.img, bufs->tgt_bufs.img,
			bufs->src_bufs.gx, bufs->src_bufs.gy, bufs->tgt_bufs.gx,
			bufs->tgt_bufs.gy, bufs->src_bufs.mask, bufs->tgt_bufs.mask,
			bufs->src_bufs.gmask, bufs->tgt_bufs.gmask, has_gmask, src.height,
			src.width, tgt.height, tgt.width, patch_size, BLACK, seed);
	}

	// copy back from device to host
	cudaCheckError(cudaMemcpy(field_ptr, bufs->field_ptr,
							  src_size * 3 * sizeof(int),
							  cudaMemcpyDeviceToHost));
}