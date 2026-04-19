#pragma once
// #define DEBUG

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
	int *field_ptr = nullptr;

	DeviceImageBuffers src_bufs;
	DeviceImageBuffers tgt_bufs;
	bool gmask_allocated = false;

	void allocate_device_buffers(int src_pixels, int tgt_pixels, bool need_gmask);

	~CudaNNFDeviceBuffers();
};