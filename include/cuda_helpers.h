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

struct curandStatePhilox4_32_10;
typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

struct CudaNNFDeviceBuffers {
	int *field_ptr = nullptr;
	curandStatePhilox4_32_10_t *rng_states = nullptr;
	int rng_capacity = 0;

	DeviceImageBuffers src_bufs;
	DeviceImageBuffers tgt_bufs;
	bool gmask_allocated = false;

	void allocate_device_buffers(int src_pixels, int tgt_pixels, bool need_gmask);

	~CudaNNFDeviceBuffers();
};

struct CudaFusedNNFDeviceBuffers {
    int *field_ptr_s2t = nullptr;  // source -> target NNF field
    int *field_ptr_t2s = nullptr;  // target -> source NNF field
	
	curandStatePhilox4_32_10_t *rng_states_s2t = nullptr;
	 curandStatePhilox4_32_10_t *rng_states_t2s = nullptr;
    int rng_capacity_s2t = 0;
    int rng_capacity_t2s = 0;

    int field_capacity_s2t = 0;
    int field_capacity_t2s = 0;

    DeviceImageBuffers a_bufs;  // image A on device
    DeviceImageBuffers b_bufs;  // image B on device
    bool gmask_allocated = false;

    void allocate_device_buffers(int a_pixels, int b_pixels, bool need_gmask);

    ~CudaFusedNNFDeviceBuffers();
};

extern "C" void launch_nnf_minimize(CudaNNFDeviceBuffers *bufs, int *field_ptr,
									const HostImageBuffers &src,
									const HostImageBuffers &tgt, bool has_gmask,
									int patch_size, int nr_pass,
									unsigned int random_seed);

extern "C" void launch_fused_nnf_minimize(
	CudaFusedNNFDeviceBuffers *bufs, int *field_s2t_host, int *field_t2s_host,
	const HostImageBuffers &a, const HostImageBuffers &b, bool has_gmask,
	int patch_size, int nr_pass, unsigned int random_seed);