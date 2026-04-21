#pragma once
#define DEBUG

void cuda_device_sync();

#ifdef DEBUG
#define LOG(fmt, ...) printf("[DEBUG] " fmt, ##__VA_ARGS__)
#else
#define LOG(fmt, ...)
#endif

struct HostImageBuffers {
	const unsigned char *img;
	const unsigned char *gx;
	const unsigned char *gy;
	const unsigned char *mask;
	const unsigned char *gmask;
	int height;
	int width;
};

struct DeviceImageBuffers {
	unsigned char *img = nullptr;
	unsigned char *gx = nullptr;
	unsigned char *gy = nullptr;
	unsigned char *mask = nullptr;
	unsigned char *gmask = nullptr;

	int pixel_capacity = 0;

	void allocate_buffers(int pixels, bool has_gmask);

	~DeviceImageBuffers();
};

struct CudaNNFDeviceBuffers {
	// ping pong field buffers for the source to target NNF
	int *s2t_curr = nullptr;
	int *s2t_prev = nullptr;
	int s2t_capacity = 0;

	// ping pong field buffers for the target to source NNF
	int *t2s_curr = nullptr;
	int *t2s_prev = nullptr;
	int t2s_capacity = 0;

	DeviceImageBuffers src_bufs;
	DeviceImageBuffers tgt_bufs;
	bool gmask_allocated = false;

	void allocate_device_buffers(int src_pixels, int tgt_pixels,
								 bool need_gmask);

	void ensure_s2t_fields(int pixels);
	void ensure_t2s_fields(int pixels);

	void swap_s2t_fields() {
		int *tmp = s2t_curr;
		s2t_curr = s2t_prev;
		s2t_prev = tmp;
	}
	void swap_t2s_fields() {
		int *tmp_buf = t2s_curr;
		t2s_curr = t2s_prev;
		t2s_prev = tmp_buf;
	}

	~CudaNNFDeviceBuffers();
};