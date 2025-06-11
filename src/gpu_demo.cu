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
#define BH_POSITION_X 30.0f
#define BH_POSITION_Y 0.0f
#define BH_POSITION_Z 0.0f
#define BH_RSCHWARZSCHILD_RADIUS 2.0f
#define EVENT_HORIZON_RADIUS 2.0f
#define INTEGRATION_DISTANCE_MULTIPLIER 50.0f
#define NUM_INTEGRATION_STEPS 1000
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
#define DISK_MAX_BRIGHTNESS 5.0f
// RayTraceResult for device

// Camera struct for device
#define BLOOM_FACTOR 8.0f
#define PLANET_RADIUS 10.0f
#define PLANET_POSITION_X 0.0f
#define PLANET_POSITION_Y 0.0f
#define PLANET_POSITION_Z 0.0f

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
__device__ float getDiskProceduralColorFactor(const Vec3f& point_on_disk_plane, const float t) {
    float dx = point_on_disk_plane.x;
    float dy = point_on_disk_plane.y;
    float distance = sqrtf(dx*dx + dy*dy);

    if (distance < DISK_INNER_RADIUS || distance > DISK_OUTER_RADIUS) {
        return 0.0f;
    }
    if (distance < EPSILON_BH) distance = EPSILON_BH;

    float theta = atan2f(dy, dx);
    float r_tex_normalized = distance/DISK_OUTER_RADIUS * 5;
    float brightness = 0.0f;
    int N = 80;
    for (int i = 1; i <= N; i++) {
        float amp_factor = 1 + sinf(i*i + theta + t);
        amp_factor /= sqrtf(i);
        float brightness_i = DISK_MAX_BRIGHTNESS * amp_factor * sinf(i*r_tex_normalized + i*i*i + theta + t);
        brightness += brightness_i;
    }
    brightness /= sqrtf(N);
    brightness = brightness * brightness + 0.4f;
    brightness *= 1/r_tex_normalized;
    
    return fmaxf(0.0f, brightness);
}

__device__ float SDF(const Vec3f& point) {
    Vec3f p = point - Vec3f(PLANET_POSITION_X, PLANET_POSITION_Y, PLANET_POSITION_Z);
    float half_side = PLANET_RADIUS / 2.0f;
    Vec3f q = Vec3f(fabsf(p.x), fabsf(p.y), fabsf(p.z)) - Vec3f(half_side, half_side, half_side);
    
    Vec3f q_max0(fmaxf(q.x, 0.0f), fmaxf(q.y, 0.0f), fmaxf(q.z, 0.0f));
    float outside_dist = q_max0.length();

    float inside_dist = fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
    
    return outside_dist + inside_dist;
}

__device__ Vec3f getPlanetColor(const Vec3f& point) {
    float dx = point.x - PLANET_POSITION_X;
    float dy = point.y - PLANET_POSITION_Y;
    float dz = point.z - PLANET_POSITION_Z;
    float distance = sqrtf(dx*dx + dy*dy + dz*dz);
    return Vec3f(1.0f, 1.0f, 1.0f);
}

__device__ Vec3f getSDFGradient(const Vec3f& point) {
    const float h = 1e-4f; // A small epsilon for the offset

    const Vec3f v1 = Vec3f(1.0f, -1.0f, -1.0f);
    const Vec3f v2 = Vec3f(-1.0f, -1.0f, 1.0f);
    const Vec3f v3 = Vec3f(-1.0f, 1.0f, -1.0f);
    const Vec3f v4 = Vec3f(1.0f, 1.0f, 1.0f);

    float sdf1 = SDF(point + v1 * h);
    float sdf2 = SDF(point + v2 * h);
    float sdf3 = SDF(point + v3 * h);
    float sdf4 = SDF(point + v4 * h);

    Vec3f gradient = (v1 * sdf1 + v2 * sdf2 + v3 * sdf3 + v4 * sdf4).normalize();
    return gradient;
}

// Device: ray tracing near black hole
__device__
RayTraceResult traceRayNearBlackHole(const Vec3f& rayOrigin, const Vec3f& rayDir, const float current_time) {
    RayTraceResult result;
    result.finalDir = rayDir;
    result.hitEventHorizon = false;
    // Initialize hitDisk and accumulatedColor
    result.hitDisk = false;
    result.hitPlanet = false;
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
    const Vec3f DISK_X_AXIS = DISK_NORMAL_VEC.cross(Vec3f(1.0f, 0.0f, 0.0f)).normalize();
    const Vec3f DISK_Y_AXIS = DISK_NORMAL_VEC.cross(DISK_X_AXIS).normalize();

    for (int i = 0; i < NUM_INTEGRATION_STEPS; ++i) {
        ds_step = (2.0f * s_max_integration_dist) / (float)NUM_INTEGRATION_STEPS;

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
        float sdf_value = SDF(current_pos_world);
        ds_step = fminf(ds_step, sdf_value);
        if (sdf_value < 0.001f) {
            result.hitPlanet = true;
            result.planetAlbedo = getPlanetColor(current_pos_world);
            Vec3f light_color(DISK_BASE_COLOR_R, DISK_BASE_COLOR_G, DISK_BASE_COLOR_B);
            Vec3f bg_color(1.0f, 1.0f, 1.0f);   
            Vec3f planet_normal = getSDFGradient(current_pos_world);
            Vec3f to_BH = BH_POSITION - current_pos_world;
            Vec3f to_BH_normalized = to_BH.normalize();
            Vec3f reflected_dir = rayDir - (planet_normal * 2.0f * rayDir.dot(planet_normal));
            reflected_dir = reflected_dir.normalize();
            float specular_factor = to_BH_normalized.dot(reflected_dir);
            specular_factor = fmaxf(0.0f, specular_factor);
            specular_factor = powf(specular_factor, 10.0f);
            float diffuse_factor = to_BH_normalized.dot(planet_normal) * 0.2f;
            diffuse_factor = fmaxf(0.0f, diffuse_factor);
            float ambient_factor = 0.1f;
            result.accumulatedColor = result.accumulatedColor + result.planetAlbedo * (light_color * (diffuse_factor + specular_factor) + bg_color * ambient_factor);
            return result;
        }
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
                Vec3f disk_project = point_on_disk_plane_world - BH_POSITION;
                float disk_brightness_factor = getDiskProceduralColorFactor(Vec3f(disk_project.dot(DISK_X_AXIS), disk_project.dot(DISK_Y_AXIS), 0.0f), current_time);
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
                    result.accumulatedColor = result.accumulatedColor + DISK_BASE_COLOR_VEC * disk_brightness_factor * (1.0f / (fabsf(dot_factor) + 0.1f));
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
    float current_time,
    const unsigned char* backgroundData,
    int backgroundWidth,
    int backgroundHeight,
    unsigned char* outputBackground, float* outputForeground,
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
    RayTraceResult traceResult = traceRayNearBlackHole(camera.position, initialRayDirection, current_time);

    int pix_idx = (j * outputWidth + i) * 3;
    Vec3f finalColorBack(0.0f, 0.0f, 0.0f); // Initialize to black, as per CPU logic
    Vec3f finalColorFore(0.0f, 0.0f, 0.0f);

    if (traceResult.hitDisk || traceResult.hitPlanet) {
        // Case 1: Disk was hit — foreground only
        finalColorFore = traceResult.accumulatedColor;
        // finalColorBack = Vec3f(0.0f, 0.0f, 0.0f); // mask background
    } 
    if (!traceResult.hitEventHorizon && !traceResult.hitPlanet) { 
        unsigned char bgRgb[3];
        getPixelFromDirection(backgroundData, backgroundWidth, backgroundHeight, traceResult.finalDir, bgRgb);

        finalColorBack.x += static_cast<float>(bgRgb[0]) / 255.0f;
        finalColorBack.y += static_cast<float>(bgRgb[1]) / 255.0f;
        finalColorBack.z += static_cast<float>(bgRgb[2]) / 255.0f;
    }

    // Clamp and set output color
    outputBackground[pix_idx + 0] = static_cast<unsigned char>(clampf(finalColorBack.x, 0.0f, 1.0f) * 255.0f);
    outputBackground[pix_idx + 1] = static_cast<unsigned char>(clampf(finalColorBack.y, 0.0f, 1.0f) * 255.0f);
    outputBackground[pix_idx + 2] = static_cast<unsigned char>(clampf(finalColorBack.z, 0.0f, 1.0f) * 255.0f);

    outputForeground[pix_idx + 0] = finalColorFore.x;
    outputForeground[pix_idx + 1] = finalColorFore.y;
    outputForeground[pix_idx + 2] = finalColorFore.z;
}

// Host: kernel launcher
void launchRenderPerspective(
    const Camera& camera,
    float current_time,
    const unsigned char* d_backgroundData,
    int backgroundWidth,
    int backgroundHeight,
    unsigned char* d_outputImageBack, float* d_outputImageFore, 
    int outputWidth,
    int outputHeight
) {
    int totalPixels = outputWidth * outputHeight;
    int threadsPerBlock = 256;
    int numBlocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    renderPerspectiveKernel<<<numBlocks, threadsPerBlock>>>(
        camera,
        current_time,
        d_backgroundData,
        backgroundWidth,
        backgroundHeight,
        d_outputImageBack,
        d_outputImageFore,
        outputWidth,
        outputHeight
    );

    cudaDeviceSynchronize();
}

// Kernel for adding fore and back 
__global__
void addOutputsKernel(unsigned char* outputImage, unsigned char* outputBackground, float *outputForeground, 
                    int outputWidth, int outputHeight) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = outputWidth * outputHeight;
    if (idx >= totalPixels) return;

    int i = idx % outputWidth;
    int j = idx / outputWidth;
    int pix_idx = (j * outputWidth + i) * 3;

    float val0 = (float)outputBackground[pix_idx + 0] + outputForeground[pix_idx + 0]*256.0;
    float val1 = (float)outputBackground[pix_idx + 1] + outputForeground[pix_idx + 1]*256.0;
    float val2 = (float)outputBackground[pix_idx + 2] + outputForeground[pix_idx + 2]*256.0;

    outputImage[pix_idx + 0] = static_cast<unsigned char>(clampf(val0, 0.0f, 255.0f));
    outputImage[pix_idx + 1] = static_cast<unsigned char>(clampf(val1, 0.0f, 255.0f));
    outputImage[pix_idx + 2] = static_cast<unsigned char>(clampf(val2, 0.0f, 255.0f));

}

void launchAddOutputsKernel(unsigned char* outputImage, unsigned char* outputBackground, float *outputForeground, 
                    int outputWidth, int outputHeight) {
    
    int totalPixels = outputWidth * outputHeight;
    int threadsPerBlock = 256;
    int numBlocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    addOutputsKernel<<<numBlocks, threadsPerBlock>>>(
        outputImage, outputBackground, outputForeground,
        outputWidth, outputHeight
    );
    cudaDeviceSynchronize();
}

// Bloom stuff 
__global__
void computeBloomWeights(const float* foreground, float* weights, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    if (idx >= totalPixels) return;

    int pix_idx = idx * 3;

    float r = foreground[pix_idx + 0]; 
    float g = foreground[pix_idx + 1];
    float b = foreground[pix_idx + 2];

    float intensity = sqrtf(0.2126*r + 0.7152*g + 0.0722*b); // or use 0.2126*r + 0.7152*g + 0.0722*b for perceptual
    weights[idx] = intensity;
}

void launchComputeBloomWeightsKernel(const float* foreground, float* weights, int width, int height) {
    int totalPixels = width * height;
    int threadsPerBlock = 256;
    int numBlocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
    computeBloomWeights<<<numBlocks, threadsPerBlock>>>(foreground, weights, width, height);
    cudaDeviceSynchronize();
}

// __global__
// void applyBloomBlurKernel(
//     const unsigned char* input,  // Foreground image
//     const float* weights,        // Per-pixel intensity weights
//     unsigned char* output,       // Blurred output
//     int width,
//     int height,
//     float bloom_strength
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int totalPixels = width * height;
//     if (idx >= totalPixels) return;

//     int x = idx % width;
//     int y = idx / width;

//     const int radius = 5; // 5x5 box blur

//     float sumr = 0.0f, sumg = 0.0f, sumb = 0.0f;
//     int count = 0;

//     for (int dy = -radius; dy <= radius; ++dy) {
//         for (int dx = -radius; dx <= radius; ++dx) {
//             int nx = x + dx;
//             int ny = y + dy;
//             if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
//                 int nidx = (ny * width + nx) * 3;
//                 sumr += (float)input[nidx + 0];
//                 sumg += (float)input[nidx + 1];
//                 sumb += (float)input[nidx + 2];
//                 count++;
//             }
//         }
//     }

//     float avg_r = sumr / (float)count;
//     float avg_g = sumg / (float)count;
//     float avg_b = sumb / (float)count;

//     float scale = (weights[idx] > 0.6f) ? weights[idx] : 0.0f;
//     scale = clampf(scale, 0.0f, 1.0f);  // Optional clamp for safety

//     int out_idx = idx * 3;
//     float orig_r = (float)input[out_idx + 0];
//     float orig_g = (float)input[out_idx + 1];
//     float orig_b = (float)input[out_idx + 2];

//     output[out_idx + 0] = static_cast<unsigned char>(clampf(orig_r + avg_r * scale * bloom_strength, 0.0f, 255.0f));
//     output[out_idx + 1] = static_cast<unsigned char>(clampf(orig_g + avg_g * scale * bloom_strength, 0.0f, 255.0f));
//     output[out_idx + 2] = static_cast<unsigned char>(clampf(orig_b + avg_b * scale * bloom_strength, 0.0f, 255.0f));

// }


// void launchBloomKernel(const unsigned char* input, const float* weights, unsigned char* output, int width, int height, float bloom_strength) {
//     int totalPixels = width * height;
//     int threadsPerBlock = 256;
//     int numBlocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

//     applyBloomBlurKernel<<<numBlocks, threadsPerBlock>>>(input, weights, output, width, height, bloom_strength);
//     cudaDeviceSynchronize();
// }

__global__
void applyBloomKernel(const float* foreground, const float* weights, float *accum, 
                    int width, int height, float maxRadius) 
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    int x = idx % width;
    int y = idx / width;

    float strength = weights[idx];
    if (strength < 1e-4f) return;

    int radius = int(BLOOM_FACTOR * strength);
    if (radius < 1) radius = 1;

    int inputIdx = idx * 3;
    float r = foreground[inputIdx + 0];
    float g = foreground[inputIdx + 1];
    float b = foreground[inputIdx + 2];
    // float norm_factor = (radius * (radius + 1) * (2*radius + 1)) / 6.0f;

    float norm = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int manhattan = abs(dx) + abs(dy);
            if (manhattan > radius) continue;
            norm += 1.0f - ((float)manhattan / (float)radius);
        }
    }

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int manhattan = abs(dx) + abs(dy);
            if (manhattan > radius) continue;

            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            float weight = (1.0f - ((float)manhattan / (float)radius))/norm;
            if (weight < 0.0f) continue;

            int outIdx = (ny * width + nx) * 3;
            atomicAdd(&accum[outIdx + 0], r * weight);
            atomicAdd(&accum[outIdx + 1], g * weight);
            atomicAdd(&accum[outIdx + 2], b * weight);
        }
    }

    // Instead try star shaped blur 

}

__global__
void horizontalBloom(const float* foreground, const float* weights,
                     float* accum, int width, int height, float maxRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int x = idx % width;
    int y = idx / width;

    float strength = weights[idx];
    if (strength < 1e-3f) return;

    int radius = max(int(maxRadius * strength), 1);
    float norm = radius + 1.0f; // ∑(1 - |i|/r) from -r to r

    int inIdx = idx * 3;
    float r = foreground[inIdx + 0];
    float g = foreground[inIdx + 1];
    float b = foreground[inIdx + 2];

    for (int dx = -radius; dx <= radius; ++dx) {
        int nx = x + dx;
        if (nx < 0 || nx >= width) continue;

        float weight = (1.0f - (abs(dx) / (float)radius)) / norm;
        int outIdx = (y * width + nx) * 3;

        atomicAdd(&accum[outIdx + 0], r * weight);
        atomicAdd(&accum[outIdx + 1], g * weight);
        atomicAdd(&accum[outIdx + 2], b * weight);
    }
}


__global__
void verticalBloom(const float* foreground, const float* weights,
                   float* output, int width, int height, float maxRadius) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int x = idx % width;
    int y = idx / width;

    float strength = weights[idx];
    if (strength < 1e-3f) return;

    int radius = max(int(maxRadius * strength), 1);
    float norm = radius + 1.0f;

    int inIdx = idx * 3;
    float r = foreground[inIdx + 0];
    float g = foreground[inIdx + 1];
    float b = foreground[inIdx + 2];

    for (int dy = -radius; dy <= radius; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= height) continue;

        float weight = (1.0f - (abs(dy) / (float)radius)) / norm;
        int outIdx = (ny * width + x) * 3;

        atomicAdd(&output[outIdx + 0], r * weight);
        atomicAdd(&output[outIdx + 1], g * weight);
        atomicAdd(&output[outIdx + 2], b * weight);
    }
}



// __global__
// void clampFloatImageToUchar(const float* input, unsigned char* output, int totalPixels) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= totalPixels * 3) return;

//     output[idx] = static_cast<unsigned char>(clampf(input[idx], 0.0f, 255.0f));
// }

void launchTwoStepBloom(const float* foreground, float *bloomed_foreground, const float* weights,
                        float* accum, int width, int height, float maxRadius) {
    int totalPixels = width * height;
    int threadsPerBlock = 256;
    int numBlocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    horizontalBloom<<<numBlocks, threadsPerBlock>>>(foreground, weights, accum, width, height, maxRadius);
    verticalBloom<<<numBlocks, threadsPerBlock>>>(accum, weights, bloomed_foreground, width, height, maxRadius);

    cudaDeviceSynchronize();
}



void launchManhattanBlurKernel(
    const float* input,
    const float* strengths,
    float* output,
    int width,
    int height,
    int maxRadius
) {
    int totalPixels = width * height;

    int threadsPerBlock = 256;
    int numBlocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    applyBloomKernel<<<numBlocks, threadsPerBlock>>>(
        input, strengths, output, width, height, maxRadius
    );
    cudaDeviceSynchronize();

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
    const float frame_rate = 24.0f;
    const float orbit_radius = 80.0f;
    const float camera_z_offset = 0.0f;

    Vec3f worldUpVector(0.0f, 0.0f, 1.0f);
    float fieldOfViewY = 75.0f;
    int outputWidth = 1000;
    int outputHeight = 1000;

    char filename_buffer[256];

    // Allocate device memory for background image
    unsigned char* d_backgroundData;
    size_t bg_bytes = bg_width * bg_height * 3 * sizeof(unsigned char);
    cudaMalloc(&d_backgroundData, bg_bytes);
    cudaMemcpy(d_backgroundData, backgroundImageData.data(), bg_bytes, cudaMemcpyHostToDevice);

    // Allocate device memory for output image (from render pass)
    unsigned char* d_outputImageBack; // This will hold the result of renderPerspectiveKernel backgorund
    float* d_outputImageFore;
    unsigned char* d_outputImage;
    float *d_bloomWeights;
    float *bloomed_foreground;

    size_t out_bytes = outputWidth * outputHeight * 3 * sizeof(unsigned char);
    size_t fore_bytes = outputWidth * outputHeight * 3 * sizeof(float);
    size_t weight_bytes = outputWidth * outputHeight * sizeof(float);

    cudaMalloc(&d_outputImageBack, out_bytes);
    cudaMalloc(&d_outputImageFore, fore_bytes);
    cudaMalloc(&d_outputImage, out_bytes); 

    cudaMalloc(&d_bloomWeights, weight_bytes);
    cudaMalloc(&bloomed_foreground, fore_bytes);




    std::vector<unsigned char> renderedImage(outputWidth * outputHeight * 3);

    std::cout << "Starting animation rendering: " << num_frames << " frames." << std::endl;
    std::cout << "Frames will be saved in: " << frame_output_dir << "/" << std::endl;

    for (int frame = 0; frame < num_frames; ++frame) {
        cudaMemset (bloomed_foreground, 0, fore_bytes );
        float angle =  2*PI * static_cast<float>(frame) / static_cast<float>(num_frames);

        Vec3f cameraPosition(
            BH_POSITION_X + orbit_radius * std::cos(angle) + 10,
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

        float current_time = static_cast<float>(frame) / frame_rate;
        // Launch CUDA kernel
        launchRenderPerspective(
            camera,
            current_time,
            d_backgroundData,
            bg_width,
            bg_height,
            d_outputImageBack, // Output of render kernel 1
            d_outputImageFore,
            outputWidth,
            outputHeight
        );

        launchComputeBloomWeightsKernel(d_outputImageFore, d_bloomWeights, outputWidth, outputHeight);

        // launchManhattanBlurKernel(d_outputImageFore, d_bloomWeights, bloomed_foreground, outputWidth, outputHeight, 12.0f);
        launchTwoStepBloom(d_outputImageFore, bloomed_foreground, d_bloomWeights, bloomed_foreground, outputWidth, outputHeight, BLOOM_FACTOR);

        launchAddOutputsKernel(d_outputImage, d_outputImageBack, bloomed_foreground, outputWidth, outputHeight); 
        
        cudaMemcpy(renderedImage.data(), d_outputImage, out_bytes, cudaMemcpyDeviceToHost);

        saveBMP(filename_buffer, renderedImage, outputWidth, outputHeight);
    }

    cudaFree(d_backgroundData);
    cudaFree(d_outputImageBack);
    cudaFree(d_outputImageFore);
    cudaFree(d_bloomWeights);

    std::cout << "Animation rendering finished." << std::endl;
    std::cout << "You can now use a tool like ffmpeg to create a video from the frame_xxxx.bmp files." << std::endl;
    std::cout << "Example ffmpeg command:" << std::endl;
    std::cout << "ffmpeg -framerate 24 -i frame_%04d.bmp -c:v libx264 -pix_fmt yuv420p output_video.mp4" << std::endl;

    return 0;
}