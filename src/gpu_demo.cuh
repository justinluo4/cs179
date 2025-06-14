#ifndef GPU_DEMO_CUH
#define GPU_DEMO_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Vec3f struct
struct Vec3f {
    float x, y, z;
    __host__ __device__ Vec3f(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3f operator+(const Vec3f& other) const { return Vec3f(x + other.x, y + other.y, z + other.z); }
    __host__ __device__ Vec3f operator-(const Vec3f& other) const { return Vec3f(x - other.x, y - other.y, z - other.z); }
    __host__ __device__ Vec3f operator*(const Vec3f& other) const { return Vec3f(x * other.x, y * other.y, z * other.z); }
    __host__ __device__ Vec3f operator*(float scalar) const { return Vec3f(x * scalar, y * scalar, z * scalar); }
    __host__ __device__ float dot(const Vec3f& other) const { return x * other.x + y * other.y + z * other.z; }
    __host__ __device__ Vec3f cross(const Vec3f& other) const {
        return Vec3f(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
    __host__ __device__ Vec3f normalize() const {
        float l = length();
        if (l == 0.0f) return Vec3f(0.0f, 0.0f, 0.0f);
        return Vec3f(x / l, y / l, z / l);
    }
};

struct RayTraceResult {
    Vec3f finalDir;
    bool hitEventHorizon;
    Vec3f accumulatedColor; // Accumulated color of the ray
    bool hitDisk; // True if the ray entered the accretion disk
    bool hitPlanet;
    Vec3f planetAlbedo;     // Color from planet texture
    float intersectionDistance; // Stores closest t_hit for opaque objects
    __host__ __device__ RayTraceResult() : finalDir(0,0,0), hitEventHorizon(false), accumulatedColor(0,0,0), hitDisk(false) {}
};

struct Camera {
    Vec3f position;
    Vec3f lookAt;
    Vec3f worldUp;
    float fovY_degrees;
    __host__ __device__ Camera() : position(), lookAt(), worldUp(), fovY_degrees(0.0f) {}
    __host__ __device__ Camera(Vec3f pos, Vec3f target, Vec3f up, float fov)
        : position(pos), lookAt(target), worldUp(up), fovY_degrees(fov) {}
};

// Constants previously declared here are now handled by #defines in gpu_demo.cu
// extern __constant__ Vec3f BH_POSITION; // Removed
// extern __constant__ float BH_RSCHWARZSCHILD_RADIUS; // Removed
// extern __constant__ float EVENT_HORIZON_RADIUS; // Removed
// extern __constant__ float INTEGRATION_DISTANCE_MULTIPLIER; // Removed
// extern __constant__ int NUM_INTEGRATION_STEPS; // Removed
// extern __constant__ float EPSILON_BH; // Removed
// extern __constant__ float PI; // Removed

// Device function helpers declarations
__device__ RayTraceResult traceRayNearBlackHole(const Vec3f& rayOrigin, const Vec3f& rayDir);

__device__ void getPixelFromDirection(const unsigned char* imageData, int imageWidth, int imageHeight,
                                        const Vec3f& direction, unsigned char* out_rgb
);

// Kernel
__global__ void renderPerspectiveKernel(Camera camera, const unsigned char* backgroundData, int backgroundWidth,
                        int backgroundHeight, unsigned char* outputImage, int outputWidth, int outputHeight
);

// runner 
void launchRenderPerspective(const Camera& camera, const unsigned char* backgroundData, int backgroundWidth,
                int backgroundHeight, unsigned char* outputImage, int outputWidth, int outputHeight
);

// Bloom effect kernels and launcher
__global__ void extractBrightPassKernel(const unsigned char* inputImage, unsigned char* brightPassImage, 
                                    int width, int height, float threshold);

__global__ void gaussianBlurKernel(const unsigned char* inputImage, unsigned char* outputImage, 
                                 int width, int height, bool horizontalPass, 
                                 const float* blurKernel_d, int blurRadius);

__global__ void additiveBlendKernel(const unsigned char* originalImage, const unsigned char* bloomImage, 
                                unsigned char* outputImage, int width, int height, float bloomIntensity);

void launchBloomEffect(unsigned char* d_renderedImage, 
                       unsigned char* d_brightPassImage, 
                       unsigned char* d_tempBlurImage, 
                       unsigned char* d_finalBlurredImage, 
                       unsigned char* d_finalOutputImage, // This will be the image with bloom + original
                       int width, int height, 
                       float brightnessThreshold, 
                       int blurRadius, 
                       const float* d_blurKernel, 
                       float bloomIntensity);

#endif 
