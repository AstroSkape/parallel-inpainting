#include <cuda_runtime.h>

#define RED 0
#define BLACK 1

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

__device__ int
compute_patch_dist(const unsigned char *src_img, const unsigned char *tgt_img,
				   const unsigned char *src_gx, const unsigned char *src_gy,
				   const unsigned char *tgt_gx, const unsigned char *tgt_gy,
				   const unsigned char *src_mask, const unsigned char *tgt_mask,
				   const unsigned char *src_gmask,
				   const unsigned char *tgt_gmask, bool has_gmask, int ys,
				   int xs, int yt, int xt, int src_h, int src_w, int tgt_h,
				   int tgt_w, int patch_size) {
	long double distance = 0;
	long double wsum = 0;
	long double kSSDScale = 9 * 255 * 255;
	long double kDistanceScale = 65535;

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
		int newy = p_x + directions[i][0];
		int newx = p_y + directions[i][1];

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
								  rand_range(seed + idx * 1337u + step * 7919u, -random_scale, random_scale),
							  0, tgt_h - 1);
		int xp = device_clamp(nnf_best_x +
								  rand_range(seed + idx * 1337u + step * 7919u, -random_scale, random_scale),
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
	}

	nnf_field[0] = nnf_best_y;
	nnf_field[1] = nnf_best_x;
	nnf_field[2] = nnf_best_d;
}

extern "C" void launch_nnf_minimize(
	int *field_ptr, const unsigned char *src_img, const unsigned char *tgt_img,
	const unsigned char *src_gx, const unsigned char *src_gy,
	const unsigned char *tgt_gx, const unsigned char *tgt_gy,
	const unsigned char *src_mask, const unsigned char *tgt_mask,
	const unsigned char *src_gmask, const unsigned char *tgt_gmask,
	bool has_gmask, int src_height, int src_width, int tgt_height,
	int tgt_width, int patch_size, int nr_pass, unsigned int random_seed) {
	// create device pointers
	int *d_field_ptr;
	unsigned char *d_src_img, *d_tgt_img;
	unsigned char *d_src_gx, *d_tgt_gx, *d_src_gy, *d_tgt_gy;
	unsigned char *d_src_mask, *d_tgt_mask;
	unsigned char *d_src_gmask = nullptr, *d_tgt_gmask = nullptr;

	int src_size = src_height * src_width;
	int tgt_size = tgt_height * tgt_width;

	cudaMalloc(&d_field_ptr, src_size * 3 * sizeof(int));
	// three channels
	cudaMalloc(&d_src_img, src_size * 3);
	cudaMalloc(&d_tgt_img, tgt_size * 3);
	cudaMalloc(&d_src_gx, src_size * 3);
	cudaMalloc(&d_tgt_gx, tgt_size * 3);
	cudaMalloc(&d_src_gy, src_size * 3);
	cudaMalloc(&d_tgt_gy, tgt_size * 3);
	// single channel
	cudaMalloc(&d_src_mask, src_size);
	cudaMalloc(&d_tgt_mask, tgt_size);
	if (has_gmask) {
		cudaMalloc(&d_src_gmask, src_size);
		cudaMalloc(&d_tgt_gmask, tgt_size);
	}

	// copy from host to device
	cudaMemcpy(d_field_ptr, field_ptr, src_size * 3 * sizeof(int),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(d_src_img, src_img, src_size * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tgt_img, tgt_img, tgt_size * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_src_gx, src_gx, src_size * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_src_gy, src_gy, src_size * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tgt_gx, tgt_gx, tgt_size * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tgt_gy, tgt_gy, tgt_size * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_src_mask, src_mask, src_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tgt_mask, tgt_mask, tgt_size, cudaMemcpyHostToDevice);
	if (has_gmask) {
		cudaMemcpy(d_src_gmask, src_gmask, src_size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_tgt_gmask, tgt_gmask, tgt_size, cudaMemcpyHostToDevice);
	}

	int num_threads = 256;
	int blocks = (src_size + num_threads - 1) / num_threads;

	for (int i = 0; i < nr_pass; i++) {
		unsigned int seed = random_seed + i * 12345u;

		// call kernel on red pixels
		nnf_minimize_kernel<<<blocks, num_threads>>>(
			d_field_ptr, d_src_img, d_tgt_img, d_src_gx, d_src_gy, d_tgt_gx,
			d_tgt_gy, d_src_mask, d_tgt_mask, d_src_gmask, d_tgt_gmask,
			has_gmask, src_height, src_width, tgt_height, tgt_width, patch_size,
			RED, seed);
		cudaDeviceSynchronize();
		// call kernel on black pixels
		nnf_minimize_kernel<<<blocks, num_threads>>>(
			d_field_ptr, d_src_img, d_tgt_img, d_src_gx, d_src_gy, d_tgt_gx,
			d_tgt_gy, d_src_mask, d_tgt_mask, d_src_gmask, d_tgt_gmask,
			has_gmask, src_height, src_width, tgt_height, tgt_width, patch_size,
			BLACK, seed);
		cudaDeviceSynchronize();
	}

	// copy back from device to host
	cudaMemcpy(field_ptr, d_field_ptr, src_size * 3 * sizeof(int),
			   cudaMemcpyDeviceToHost);

	// Free device pointers
	cudaFree(d_field_ptr);
	cudaFree(d_src_img);
	cudaFree(d_tgt_img);
	cudaFree(d_src_gx);
	cudaFree(d_tgt_gx);
	cudaFree(d_src_gy);
	cudaFree(d_tgt_gy);
	cudaFree(d_src_mask);
	cudaFree(d_tgt_mask);
	if (has_gmask) {
		cudaFree(d_src_gmask);
		cudaFree(d_tgt_gmask);
	}
}