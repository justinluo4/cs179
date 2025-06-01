#include "gpu_demo.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include "file_io.h"

// CUDA math helpers
__device__ float clampf(float x, float a, float b) { return fminf(fmaxf(x, a), b); }

// Device version of Vec3f

// Black Hole and rendering constants
#define PI 3.14159265358979f
#define BH_POSITION_X 50.0f
#define BH_POSITION_Y 0.0f
#define BH_POSITION_Z 0.0f
#define BH_RSCHWARZSCHILD_RADIUS 2.0f
#define EVENT_HORIZON_RADIUS 2.0f
#define INTEGRATION_DISTANCE_MULTIPLIER 50.0f
#define NUM_INTEGRATION_STEPS 400
#define EPSILON_BH 1e-4f

// Accretion Disk Parameters
#define DISK_NORMAL_X 0.0f
#define DISK_NORMAL_Y 0.8f
#define DISK_NORMAL_Z 0.6f
#define DISK_INNER_RADIUS (EVENT_HORIZON_RADIUS * 3.0f)
#define DISK_OUTER_RADIUS (EVENT_HORIZON_RADIUS * 15.0f)
#define DISK_BASE_COLOR_R 1.0f
#define DISK_BASE_COLOR_G 0.4f
#define DISK_BASE_COLOR_B 0.1f
#define DISK_MAX_BRIGHTNESS 2.0f

// RayTraceResult for device

// Camera struct for device

// Device: get pixel color from direction
__device__
void getPixelFromDirection(const unsigned char* imageData, int imageWidth, int imageHeight, const Vec3f& direction, unsigned char* rgb_out) {
    float phi = atan2f(direction.y, direction.x);
    float clamped_normZ = clampf(direction.z, -1.0f, 1.0f);
    float theta = acosf(clamped_normZ);

    float u = (phi + PI) / (2.0f * PI);
    float v = theta / PI;

    int px = (int)roundf(u * imageWidth);
    int py = (int)roundf(v * imageHeight);

    if (px >= imageWidth) px = imageWidth - 1;
    if (py >= imageHeight) py = imageHeight - 1;
    if (px < 0) px = 0;
    if (py < 0) py = 0;

    int pixel_idx = (py * imageWidth + px) * 3;
    rgb_out[0] = imageData[pixel_idx + 0];
    rgb_out[1] = imageData[pixel_idx + 1];
    rgb_out[2] = imageData[pixel_idx + 2];
}

// Device: get procedural color factor for accretion disk
__device__ float getDiskProceduralColorFactor(const Vec3f& point_on_disk_plane) {
    float dx = point_on_disk_plane.x;
    float dy = point_on_disk_plane.y;
    float distance = sqrtf(dx*dx + dy*dy);

    if (distance < DISK_INNER_RADIUS || distance > DISK_OUTER_RADIUS) {
        return 0.0f;
    }
    if (distance < EPSILON_BH) distance = EPSILON_BH;

    float theta = atan2f(dy, dx);
    float r_tex_normalized = (3.1 * distance + sinf(distance) + 2*sinf(distance*2.7) + 3 * sinf(distance / 3.2)) / DISK_OUTER_RADIUS;

    float radial_factor = 1.0f - ( (distance - DISK_INNER_RADIUS) / (DISK_OUTER_RADIUS - DISK_INNER_RADIUS) );
    radial_factor = fmaxf(0.0f, fminf(1.0f, radial_factor));
    
    float angular_factor = 0.8f + 0.6f * sinf(6.0f * theta + 10.0f * r_tex_normalized * r_tex_normalized );

    float brightness = DISK_MAX_BRIGHTNESS * radial_factor * angular_factor;
    
    return fmaxf(0.0f, fminf(DISK_MAX_BRIGHTNESS, brightness));
}

// Device: ray tracing near black hole
__device__
RayTraceResult traceRayNearBlackHole(const Vec3f& rayOrigin, const Vec3f& rayDir) {
    RayTraceResult result;
    result.finalDir = rayDir;
    result.hitEventHorizon = false;
    // Initialize hitDisk and accumulatedColor
    result.hitDisk = false;
    result.accumulatedColor = Vec3f(0.0f, 0.0f, 0.0f);

    Vec3f BH_POSITION(BH_POSITION_X, BH_POSITION_Y, BH_POSITION_Z);

    Vec3f L = BH_POSITION - rayOrigin;
    float t_ca = L.dot(rayDir);
    Vec3f P_ca = rayOrigin + rayDir * t_ca;
    Vec3f b_vec_3d = P_ca - BH_POSITION;
    float b_impact_param_3d = b_vec_3d.length();

    // Vec3f O_minus_C = rayOrigin - BH_POSITION;
    // float a_quad = rayDir.dot(rayDir);
    // float b_quad = 2.0f * rayDir.dot(O_minus_C);
    // float c_quad = O_minus_C.dot(O_minus_C) - EVENT_HORIZON_RADIUS * EVENT_HORIZON_RADIUS;
    // float discriminant = b_quad * b_quad - 4.0f * a_quad * c_quad;

    // if (discriminant >= 0.0f) {
    //     float t0 = (-b_quad - sqrtf(discriminant)) / (2.0f * a_quad);
    //     float t1 = (-b_quad + sqrtf(discriminant)) / (2.0f * a_quad);
    //     if (t0 > 0 || t1 > 0) {
    //         float t_intersect = (t0 > 0 && t1 > 0) ? fminf(t0, t1) : fmaxf(t0, t1);
    //         if (t_intersect > 0) {
    //             if (b_impact_param_3d < EVENT_HORIZON_RADIUS && t_ca > 0) {
    //                 result.hitEventHorizon = true;
    //                 return result;
    //             }
    //         }
    //     }
    // }

    Vec3f y_axis_2d_plane;
    if (b_impact_param_3d < EPSILON_BH) {
        if (t_ca > 0) {
            result.hitEventHorizon = true;
            return result;
        }
        return result;
    }
    y_axis_2d_plane = b_vec_3d.normalize();
    Vec3f x_axis_2d_plane = rayDir;

    float s_max_integration_dist = INTEGRATION_DISTANCE_MULTIPLIER * BH_RSCHWARZSCHILD_RADIUS;
    if (s_max_integration_dist < b_impact_param_3d * 2.5f) s_max_integration_dist = b_impact_param_3d * 2.5f;
    if (s_max_integration_dist < 20.0f * BH_RSCHWARZSCHILD_RADIUS) s_max_integration_dist = 20.0f * BH_RSCHWARZSCHILD_RADIUS;
    if (s_max_integration_dist <= 0.0f) s_max_integration_dist = 10.0f;

    float x_pos = -s_max_integration_dist;
    float y_pos = b_impact_param_3d;

    float r_init = sqrtf(x_pos * x_pos + y_pos * y_pos);
    if (r_init <= EVENT_HORIZON_RADIUS) {
        result.hitEventHorizon = true;
        return result;
    }
    float term_rs_over_r_init = BH_RSCHWARZSCHILD_RADIUS / r_init;
    if (fabsf(1.0f - term_rs_over_r_init) < EPSILON_BH || (1.0f - term_rs_over_r_init) <= 0.0f) {
        result.hitEventHorizon = true;
        return result;
    }
    float n_init = 1.0f / (1.0f - term_rs_over_r_init);

    float px_val = n_init * 1.0f;
    float py_val = n_init * 0.0f;

    float ds_step = (2.0f * s_max_integration_dist) / (float)NUM_INTEGRATION_STEPS;
    // Initialize variables for disk intersection
    float disk_distance = 0.0f;
    float disk_distance_prev = 0.0f;
    const Vec3f DISK_NORMAL_VEC(DISK_NORMAL_X, DISK_NORMAL_Y, DISK_NORMAL_Z);
    const Vec3f DISK_BASE_COLOR_VEC(DISK_BASE_COLOR_R, DISK_BASE_COLOR_G, DISK_BASE_COLOR_B);

    for (int i = 0; i < NUM_INTEGRATION_STEPS; ++i) {
        float r_sq = x_pos * x_pos + y_pos * y_pos;
        float r_current = sqrtf(r_sq);

        if (r_current <= EVENT_HORIZON_RADIUS) {
            result.hitEventHorizon = true;
            return result;
        }

        float term_rs_over_r_current = BH_RSCHWARZSCHILD_RADIUS / r_current;
        if (fabsf(1.0f - term_rs_over_r_current) < EPSILON_BH) {
            result.hitEventHorizon = true;
            return result;
        }
        if ((1.0f - term_rs_over_r_current) <= 0.0f) {
            result.hitEventHorizon = true;
            return result;
        }
        float n_current = 1.0f / (1.0f - term_rs_over_r_current);

        if (r_current < EPSILON_BH) {
            result.hitEventHorizon = true;
            return result;
        }
        float r_cubed = r_sq * r_current;

        float accel_common_factor = -n_current * n_current * BH_RSCHWARZSCHILD_RADIUS / r_cubed;
        float dp_x = accel_common_factor * x_pos * ds_step;
        float dp_y = accel_common_factor * y_pos * ds_step;

        float dx = (px_val / n_current) * ds_step;
        float dy = (py_val / n_current) * ds_step;

        // Calculate the point in 3D space (relative to BH for disk check)
        // x_pos and y_pos are coordinates in the 2D plane defined by x_axis_2d_plane and y_axis_2d_plane,
        // with the BH at the origin of this 2D system.
        Vec3f current_pos_relative_to_BH = x_axis_2d_plane * x_pos + y_axis_2d_plane * y_pos;
        Vec3f current_pos_world = BH_POSITION + current_pos_relative_to_BH;

        disk_distance_prev = disk_distance; // Store previous distance for crossing check
        disk_distance = (current_pos_world - BH_POSITION).dot(DISK_NORMAL_VEC);

        // Check for disk intersection if ray crosses the disk plane (sign change in distance)
        if (disk_distance * disk_distance_prev <= 0.0f && i > 0) { // i > 0 to ensure disk_distance_prev is initialized
            // Point of intersection with the disk plane (approximately, can refine with line-plane intersection if needed)
            // For simplicity, use current_pos_world, assuming ds_step is small enough.
            Vec3f point_on_disk_plane_world = current_pos_world - DISK_NORMAL_VEC * disk_distance; // Project current_pos_world onto disk plane
            
            // Calculate distance from BH center to this point *in the disk plane*
            float dist_to_bh_in_disk_plane = (point_on_disk_plane_world - BH_POSITION).length(); // This is not quite right for procedural color.
                                                                                                // getDiskProceduralColorFactor expects a point *on* the disk relative to its center.
                                                                                                // The point_on_disk_plane_world IS on the disk plane.
                                                                                                // Its coordinates relative to BH_POSITION are what's needed.

            if (dist_to_bh_in_disk_plane >= DISK_INNER_RADIUS && dist_to_bh_in_disk_plane <= DISK_OUTER_RADIUS) {
                // Use current_pos_world as the point for color factor, assuming it's close enough to the actual intersection point
                // The procedural color function in CPU demo used `current_pos_3d` (which is `current_pos_world` here)
                // and `BH_POSITION` as `disk_center_on_plane`. This seems correct.
                float disk_brightness_factor = getDiskProceduralColorFactor(Vec3f(dist_to_bh_in_disk_plane, dist_to_bh_in_disk_plane, 0.0f));
                if (disk_brightness_factor > 0.0f) {
                    result.hitDisk = true; // Mark that disk was hit at least once
                    // Accumulate color. If ray passes through multiple times, this will add up.
                    // Consider if only the first hit should count or if accumulation is desired.
                    // CPU demo seems to just set hitDisk and adds to accumulatedColor once.
                    // Let's stick to accumulating if it hits multiple times, can be adjusted.
                    float tx = px_val / n_current;
                    float ty = py_val / n_current;
                    float t_len = sqrtf(tx * tx + ty * ty);
                    float vx_norm = tx / t_len;
                    float vy_norm = ty / t_len;
                    Vec3f finalDir = (x_axis_2d_plane * vx_norm + y_axis_2d_plane * vy_norm).normalize();
                    float dot_factor = finalDir.dot(DISK_NORMAL_VEC);
                    result.accumulatedColor = result.accumulatedColor + DISK_BASE_COLOR_VEC * disk_brightness_factor * (1.0f / (fabsf(dot_factor) + EPSILON_BH));
                }
            }
        }


        px_val += dp_x;
        py_val += dp_y;

        x_pos += dx;
        y_pos += dy;

        if (i > NUM_INTEGRATION_STEPS / 2) {
            if (x_pos > EPSILON_BH) {
                bool is_moving_radially_outward = (x_pos * px_val + y_pos * py_val) > 0;
                if (is_moving_radially_outward && (r_current > s_max_integration_dist * 0.90f)) {
                    break;
                }
            }
        }
        ds_step = (s_max_integration_dist*1.5f) / (float)NUM_INTEGRATION_STEPS + fabsf(disk_distance) / 10.0f;
    }

    float r_final = sqrtf(x_pos * x_pos + y_pos * y_pos);
    if (r_final <= EVENT_HORIZON_RADIUS) {
        result.hitEventHorizon = true;
        return result;
    }
    float term_rs_over_r_final = BH_RSCHWARZSCHILD_RADIUS / r_final;
    if (fabsf(1.0f - term_rs_over_r_final) < EPSILON_BH || (1.0f - term_rs_over_r_final) <= 0.0f) {
        result.hitEventHorizon = true;
        return result;
    }
    float n_final = 1.0f / (1.0f - term_rs_over_r_final);

    float final_tx = px_val / n_final;
    float final_ty = py_val / n_final;

    float final_t_len = sqrtf(final_tx * final_tx + final_ty * final_ty);
    if (final_t_len < EPSILON_BH) {
        result.finalDir = rayDir;
        return result;
    }
    float final_vx_norm = final_tx / final_t_len;
    float final_vy_norm = final_ty / final_t_len;

    result.finalDir = (x_axis_2d_plane * final_vx_norm + y_axis_2d_plane * final_vy_norm).normalize();

    return result;
}

// CUDA kernel: render perspective
__global__
void renderPerspectiveKernel(
    Camera camera,
    const unsigned char* backgroundData,
    int backgroundWidth,
    int backgroundHeight,
    unsigned char* outputImage,
    int outputWidth,
    int outputHeight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = outputWidth * outputHeight;
    if (idx >= totalPixels) return;

    int i = idx % outputWidth;
    int j = idx / outputWidth;

    Vec3f viewDir = (camera.lookAt - camera.position).normalize();
    Vec3f camRight = viewDir.cross(camera.worldUp).normalize();
    Vec3f camUp = camRight.cross(viewDir).normalize();

    float aspectRatio = (float)outputWidth / (float)outputHeight;
    float fovY_radians = camera.fovY_degrees * (PI / 180.0f);

    float viewPlaneHeight = 2.0f * tanf(fovY_radians / 2.0f);
    float viewPlaneWidth = viewPlaneHeight * aspectRatio;

    float Px = (2.0f * ((float)i + 0.5f) / (float)outputWidth - 1.0f);
    float Py = (1.0f - 2.0f * ((float)j + 0.5f) / (float)outputHeight);

    Vec3f initialRayDirection = (viewDir + camRight * Px * (viewPlaneWidth / 2.0f) + camUp * Py * (viewPlaneHeight / 2.0f)).normalize();
    RayTraceResult traceResult = traceRayNearBlackHole(camera.position, initialRayDirection);

    int pix_idx = (j * outputWidth + i) * 3;
    Vec3f finalColor(0.0f, 0.0f, 0.0f); // Initialize to black, as per CPU logic

    if (traceResult.hitDisk) {
        finalColor = traceResult.accumulatedColor; // Start with disk color
        // If disk is hit AND ray does NOT fall into event horizon, add background
        if (!traceResult.hitEventHorizon) { 
            unsigned char bgRgb[3];
            getPixelFromDirection(backgroundData, backgroundWidth, backgroundHeight, traceResult.finalDir, bgRgb);
            finalColor.x += static_cast<float>(bgRgb[0]) / 255.0f;
            finalColor.y += static_cast<float>(bgRgb[1]) / 255.0f;
            finalColor.z += static_cast<float>(bgRgb[2]) / 255.0f;
        }
        // If disk is hit AND ray hits event horizon, finalColor remains the accumulatedColor from the disk (emission before falling in).
    } else { // Disk was NOT hit
        // If disk not hit AND ray does NOT fall into event horizon, use background
        if (!traceResult.hitEventHorizon) { 
            unsigned char rgb[3];
            getPixelFromDirection(backgroundData, backgroundWidth, backgroundHeight, traceResult.finalDir, rgb);
            finalColor.x = static_cast<float>(rgb[0]) / 255.0f;
            finalColor.y = static_cast<float>(rgb[1]) / 255.0f;
            finalColor.z = static_cast<float>(rgb[2]) / 255.0f;
        }
        // If disk not hit AND ray hits event horizon, finalColor remains (0.0f, 0.0f, 0.0f) -> black from initialization.
    }

    // Clamp and set output color
    outputImage[pix_idx + 0] = static_cast<unsigned char>(clampf(finalColor.x, 0.0f, 1.0f) * 255.0f);
    outputImage[pix_idx + 1] = static_cast<unsigned char>(clampf(finalColor.y, 0.0f, 1.0f) * 255.0f);
    outputImage[pix_idx + 2] = static_cast<unsigned char>(clampf(finalColor.z, 0.0f, 1.0f) * 255.0f);
}

// Host: kernel launcher
void launchRenderPerspective(
    const Camera& camera,
    const unsigned char* d_backgroundData,
    int backgroundWidth,
    int backgroundHeight,
    unsigned char* d_outputImage,
    int outputWidth,
    int outputHeight
) {
    int totalPixels = outputWidth * outputHeight;
    int threadsPerBlock = 256;
    int numBlocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    renderPerspectiveKernel<<<numBlocks, threadsPerBlock>>>(
        camera,
        d_backgroundData,
        backgroundWidth,
        backgroundHeight,
        d_outputImage,
        outputWidth,
        outputHeight
    );
    cudaDeviceSynchronize();
}

// Bloom Effect Kernels

__global__ void extractBrightPassKernel(const unsigned char* inputImage, unsigned char* brightPassImage, 
                                    int width, int height, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        float r = static_cast<float>(inputImage[idx + 0]) / 255.0f;
        float g = static_cast<float>(inputImage[idx + 1]) / 255.0f;
        float b = static_cast<float>(inputImage[idx + 2]) / 255.0f;

        // Calculate brightness (e.g., luminance or max component)
        // Using a common luminance approximation
        float brightness = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        // Alternative: float brightness = fmaxf(r, fmaxf(g, b));

        if (brightness > threshold) {
            brightPassImage[idx + 0] = inputImage[idx + 0];
            brightPassImage[idx + 1] = inputImage[idx + 1];
            brightPassImage[idx + 2] = inputImage[idx + 2];
        } else {
            brightPassImage[idx + 0] = 0;
            brightPassImage[idx + 1] = 0;
            brightPassImage[idx + 2] = 0;
        }
    }
}

#define BLUR_BLOCK_DIM_X 16
#define BLUR_BLOCK_DIM_Y 16

__global__ void gaussianBlurKernel(const unsigned char* inputImage, unsigned char* outputImage, 
                                 int width, int height, bool horizontalPass, 
                                 const float* blurKernel_d, int blurRadius) {
    // blurKernel_d is the 1D Gaussian kernel (weights)
    // blurRadius is half the kernel width (e.g., for a 5x1 kernel, radius is 2)

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
    float weightSum = 0.0f;

    if (horizontalPass) {
        for (int k = -blurRadius; k <= blurRadius; ++k) {
            int ix = x + k;
            if (ix >= 0 && ix < width) {
                int offset = (y * width + ix) * 3;
                float weight = blurKernel_d[k + blurRadius];
                sumR += static_cast<float>(inputImage[offset + 0]) * weight;
                sumG += static_cast<float>(inputImage[offset + 1]) * weight;
                sumB += static_cast<float>(inputImage[offset + 2]) * weight;
                weightSum += weight;
            }
        }
    } else { // Vertical pass
        for (int k = -blurRadius; k <= blurRadius; ++k) {
            int iy = y + k;
            if (iy >= 0 && iy < height) {
                int offset = (iy * width + x) * 3;
                float weight = blurKernel_d[k + blurRadius];
                sumR += static_cast<float>(inputImage[offset + 0]) * weight;
                sumG += static_cast<float>(inputImage[offset + 1]) * weight;
                sumB += static_cast<float>(inputImage[offset + 2]) * weight;
                weightSum += weight;
            }
        }
    }

    int outIdx = (y * width + x) * 3;
    if (weightSum > 0) { // Avoid division by zero if all neighbors were out of bounds
        outputImage[outIdx + 0] = static_cast<unsigned char>(clampf(sumR / weightSum, 0.0f, 255.0f));
        outputImage[outIdx + 1] = static_cast<unsigned char>(clampf(sumG / weightSum, 0.0f, 255.0f));
        outputImage[outIdx + 2] = static_cast<unsigned char>(clampf(sumB / weightSum, 0.0f, 255.0f));
    } else {
        outputImage[outIdx + 0] = inputImage[(y * width + x) * 3 + 0];
        outputImage[outIdx + 1] = inputImage[(y * width + x) * 3 + 1];
        outputImage[outIdx + 2] = inputImage[(y * width + x) * 3 + 2];
    }
}

__global__ void additiveBlendKernel(const unsigned char* originalImage, const unsigned char* bloomImage, 
                                unsigned char* outputImage, int width, int height, float bloomIntensity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        
        float originalR = static_cast<float>(originalImage[idx + 0]);
        float originalG = static_cast<float>(originalImage[idx + 1]);
        float originalB = static_cast<float>(originalImage[idx + 2]);

        float bloomR = static_cast<float>(bloomImage[idx + 0]) * bloomIntensity;
        float bloomG = static_cast<float>(bloomImage[idx + 1]) * bloomIntensity;
        float bloomB = static_cast<float>(bloomImage[idx + 2]) * bloomIntensity;

        outputImage[idx + 0] = static_cast<unsigned char>(clampf(originalR + bloomR, 0.0f, 255.0f));
        outputImage[idx + 1] = static_cast<unsigned char>(clampf(originalG + bloomG, 0.0f, 255.0f));
        outputImage[idx + 2] = static_cast<unsigned char>(clampf(originalB + bloomB, 0.0f, 255.0f));
    }
}

// Host launcher for bloom effect
void launchBloomEffect(unsigned char* d_renderedImage,      // Input: original rendered image
                       unsigned char* d_brightPassImage,    // Buffer for bright pass result
                       unsigned char* d_tempBlurImage,      // Buffer for first blur pass (e.g., horizontal)
                       unsigned char* d_finalBlurredImage,  // Buffer for second blur pass (e.g., vertical)
                       unsigned char* d_finalOutputImage,   // Output: original + bloom
                       int width, int height, 
                       float brightnessThreshold, 
                       int blurRadius,              // Half-width of the blur kernel
                       const float* d_blurKernel,   // Pre-allocated and copied Gaussian kernel on device
                       float bloomIntensity) {

    dim3 threadsPerBlock(BLUR_BLOCK_DIM_X, BLUR_BLOCK_DIM_Y);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 1. Extract bright pass
    extractBrightPassKernel<<<numBlocks, threadsPerBlock>>>(d_renderedImage, d_brightPassImage, width, height, brightnessThreshold);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error after extractBrightPassKernel: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize(); // Ensure completion before next step

    // 2. Gaussian Blur - Horizontal pass
    // Input: d_brightPassImage, Output: d_tempBlurImage
    gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(d_brightPassImage, d_tempBlurImage, width, height, true, d_blurKernel, blurRadius);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error after gaussianBlurKernel (horizontal): %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    // 3. Gaussian Blur - Vertical pass
    // Input: d_tempBlurImage, Output: d_finalBlurredImage
    gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(d_tempBlurImage, d_finalBlurredImage, width, height, false, d_blurKernel, blurRadius);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error after gaussianBlurKernel (vertical): %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    // 4. Additive Blend
    // Input: d_renderedImage (original), d_finalBlurredImage (bloom), Output: d_finalOutputImage
    additiveBlendKernel<<<numBlocks, threadsPerBlock>>>(d_renderedImage, d_finalBlurredImage, d_finalOutputImage, width, height, bloomIntensity);
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error after additiveBlendKernel: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}

// Helper function to generate 1D Gaussian kernel
std::vector<float> generateGaussianKernel(int radius, float sigma) {
    int kernelSize = 2 * radius + 1;
    std::vector<float> kernel(kernelSize);
    float sum = 0.0f;
    for (int i = 0; i < kernelSize; ++i) {
        float x = static_cast<float>(i - radius);
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    // Normalize the kernel
    for (int i = 0; i < kernelSize; ++i) {
        kernel[i] /= sum;
    }
    return kernel;
}

// Host main for GPU version
int main() {
    const char* frame_output_dir = "frames_temp";

    struct stat st = {0};
    if (stat(frame_output_dir, &st) == -1) {
        if (mkdir(frame_output_dir, 0775) != 0 && errno != EEXIST) {
            std::cerr << "Error: Could not create directory " << frame_output_dir << std::endl;
            return 1;
        }
    }

    int bg_width, bg_height;
    std::vector<unsigned char> backgroundImageData = readBMP("../assets/Stellarium3.bmp", bg_width, bg_height);

    if (backgroundImageData.empty()) {
        std::cerr << "Failed to load background image. Exiting." << std::endl;
        return 1;
    }

    std::cout << "Successfully read background BMP image with dimensions: " << bg_width << "x" << bg_height << std::endl;

    // Animation Parameters
    const int num_frames = 300;
    const float orbit_radius = 60.0f;
    const float camera_z_offset = 0.0f;

    Vec3f worldUpVector(0.0f, 0.0f, 1.0f);
    float fieldOfViewY = 75.0f;
    int outputWidth = 720;
    int outputHeight = 480;

    char filename_buffer[256];

    // Allocate device memory for background image
    unsigned char* d_backgroundData;
    size_t bg_bytes = bg_width * bg_height * 3 * sizeof(unsigned char);
    cudaMalloc(&d_backgroundData, bg_bytes);
    cudaMemcpy(d_backgroundData, backgroundImageData.data(), bg_bytes, cudaMemcpyHostToDevice);

    // Allocate device memory for output image (from render pass)
    unsigned char* d_outputImage; // This will hold the result of renderPerspectiveKernel
    size_t out_bytes = outputWidth * outputHeight * 3 * sizeof(unsigned char);
    cudaMalloc(&d_outputImage, out_bytes);

    // Allocate device memory for bloom effect
    unsigned char* d_brightPassImage;
    unsigned char* d_tempBlurImage;
    unsigned char* d_finalBlurredImage;
    unsigned char* d_bloomResultImage; // Final image with bloom effect applied
    cudaMalloc(&d_brightPassImage, out_bytes);
    cudaMalloc(&d_tempBlurImage, out_bytes);
    cudaMalloc(&d_finalBlurredImage, out_bytes);
    cudaMalloc(&d_bloomResultImage, out_bytes); // This will be the target for launchBloomEffect

    // Bloom parameters
    float brightnessThreshold = 0.7f; // Example: Pixels brighter than 70% luminance
    int blurRadius = 5;               // Example: 11x11 blur kernel (radius 5)
    float blurSigma = 2.5f;           // Sigma for Gaussian
    float bloomIntensity = 1.0f;      // Intensity of the bloom effect

    // Generate Gaussian kernel on CPU
    std::vector<float> h_blurKernel = generateGaussianKernel(blurRadius, blurSigma);
    float* d_blurKernel;
    cudaMalloc(&d_blurKernel, h_blurKernel.size() * sizeof(float));
    cudaMemcpy(d_blurKernel, h_blurKernel.data(), h_blurKernel.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<unsigned char> renderedImage(outputWidth * outputHeight * 3);

    std::cout << "Starting animation rendering: " << num_frames << " frames." << std::endl;
    std::cout << "Frames will be saved in: " << frame_output_dir << "/" << std::endl;

    for (int frame = 0; frame < num_frames; ++frame) {
        float angle =  PI * static_cast<float>(frame) / static_cast<float>(num_frames);

        Vec3f cameraPosition(
            BH_POSITION_X + orbit_radius * std::cos(angle) + 20,
            BH_POSITION_Y + orbit_radius * std::sin(angle),
            camera_z_offset + 10.0f
        );
        Vec3f lookAtTarget(BH_POSITION_X, BH_POSITION_Y, BH_POSITION_Z);

        Camera camera;
        camera.position = cameraPosition;
        camera.lookAt = lookAtTarget;
        camera.worldUp = worldUpVector;
        camera.fovY_degrees = fieldOfViewY;

        snprintf(filename_buffer, sizeof(filename_buffer), "%s/frame_%04d.bmp", frame_output_dir, frame);
        std::cout << "Rendering frame " << (frame + 1) << "/" << num_frames << " to " << filename_buffer << "..." << std::endl;

        // Launch CUDA kernel
        launchRenderPerspective(
            camera,
            d_backgroundData,
            bg_width,
            bg_height,
            d_outputImage, // Output of render kernel
            outputWidth,
            outputHeight
        );

        // Apply bloom effect
        // d_outputImage is the input (original render)
        // d_bloomResultImage is the final output (original + bloom)
        launchBloomEffect(
            d_outputImage,        // d_renderedImage (original from render pass)
            d_brightPassImage,    
            d_tempBlurImage,      
            d_finalBlurredImage,  
            d_bloomResultImage,   // d_finalOutputImage (this will have original + bloom)
            outputWidth, outputHeight,
            brightnessThreshold,
            blurRadius,
            d_blurKernel,
            bloomIntensity
        );
        
        // Copy result (with bloom) back to host
        cudaMemcpy(renderedImage.data(), d_outputImage, out_bytes, cudaMemcpyDeviceToHost);

        saveBMP(filename_buffer, renderedImage, outputWidth, outputHeight);
    }

    cudaFree(d_backgroundData);
    cudaFree(d_outputImage);
    // Free bloom effect memory
    cudaFree(d_brightPassImage);
    cudaFree(d_tempBlurImage);
    cudaFree(d_finalBlurredImage);
    cudaFree(d_bloomResultImage);
    cudaFree(d_blurKernel);

    std::cout << "Animation rendering finished." << std::endl;
    std::cout << "You can now use a tool like ffmpeg to create a video from the frame_xxxx.bmp files." << std::endl;
    std::cout << "Example ffmpeg command:" << std::endl;
    std::cout << "ffmpeg -framerate 24 -i frame_%04d.bmp -c:v libx264 -pix_fmt yuv420p output_video.mp4" << std::endl;

    return 0;
}