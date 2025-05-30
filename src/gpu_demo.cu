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
#define NUM_INTEGRATION_STEPS 200
#define EPSILON_BH 1e-4f

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

// Device: ray tracing near black hole
__device__
RayTraceResult traceRayNearBlackHole(const Vec3f& rayOrigin, const Vec3f& rayDir) {
    RayTraceResult result;
    result.finalDir = rayDir;
    result.hitEventHorizon = false;

    Vec3f BH_POSITION(BH_POSITION_X, BH_POSITION_Y, BH_POSITION_Z);

    Vec3f L = BH_POSITION - rayOrigin;
    float t_ca = L.dot(rayDir);
    Vec3f P_ca = rayOrigin + rayDir * t_ca;
    Vec3f b_vec_3d = P_ca - BH_POSITION;
    float b_impact_param_3d = b_vec_3d.length();

    Vec3f O_minus_C = rayOrigin - BH_POSITION;
    float a_quad = rayDir.dot(rayDir);
    float b_quad = 2.0f * rayDir.dot(O_minus_C);
    float c_quad = O_minus_C.dot(O_minus_C) - EVENT_HORIZON_RADIUS * EVENT_HORIZON_RADIUS;
    float discriminant = b_quad * b_quad - 4.0f * a_quad * c_quad;

    if (discriminant >= 0.0f) {
        float t0 = (-b_quad - sqrtf(discriminant)) / (2.0f * a_quad);
        float t1 = (-b_quad + sqrtf(discriminant)) / (2.0f * a_quad);
        if (t0 > 0 || t1 > 0) {
            float t_intersect = (t0 > 0 && t1 > 0) ? fminf(t0, t1) : fmaxf(t0, t1);
            if (t_intersect > 0) {
                if (b_impact_param_3d < EVENT_HORIZON_RADIUS && t_ca > 0) {
                    result.hitEventHorizon = true;
                    return result;
                }
            }
        }
    }

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
    if (traceResult.hitEventHorizon) {
        outputImage[pix_idx + 0] = 0;
        outputImage[pix_idx + 1] = 0;
        outputImage[pix_idx + 2] = 0;
    } else {
        unsigned char rgb[3];
        getPixelFromDirection(backgroundData, backgroundWidth, backgroundHeight, traceResult.finalDir, rgb);
        outputImage[pix_idx + 0] = rgb[0];
        outputImage[pix_idx + 1] = rgb[1];
        outputImage[pix_idx + 2] = rgb[2];
    }
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
    std::vector<unsigned char> backgroundImageData = readBMP("../assets/galaxy.bmp", bg_width, bg_height);

    if (backgroundImageData.empty()) {
        std::cerr << "Failed to load background image. Exiting." << std::endl;
        return 1;
    }

    std::cout << "Successfully read background BMP image with dimensions: " << bg_width << "x" << bg_height << std::endl;

    // Animation Parameters
    const int num_frames = 120;
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

    // Allocate device memory for output image
    unsigned char* d_outputImage;
    size_t out_bytes = outputWidth * outputHeight * 3 * sizeof(unsigned char);
    cudaMalloc(&d_outputImage, out_bytes);

    std::vector<unsigned char> renderedImage(outputWidth * outputHeight * 3);

    std::cout << "Starting animation rendering: " << num_frames << " frames." << std::endl;
    std::cout << "Frames will be saved in: " << frame_output_dir << "/" << std::endl;

    for (int frame = 0; frame < num_frames; ++frame) {
        float angle = PI/2 + PI * static_cast<float>(frame) / static_cast<float>(num_frames);

        Vec3f cameraPosition(
            BH_POSITION_X + orbit_radius * std::cos(angle),
            BH_POSITION_Y + orbit_radius * std::sin(angle),
            camera_z_offset
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
            d_outputImage,
            outputWidth,
            outputHeight
        );

        // Copy result back to host
        cudaMemcpy(renderedImage.data(), d_outputImage, out_bytes, cudaMemcpyDeviceToHost);

        saveBMP(filename_buffer, renderedImage, outputWidth, outputHeight);
    }

    cudaFree(d_backgroundData);
    cudaFree(d_outputImage);

    std::cout << "Animation rendering finished." << std::endl;
    std::cout << "You can now use a tool like ffmpeg to create a video from the frame_xxxx.bmp files." << std::endl;
    std::cout << "Example ffmpeg command:" << std::endl;
    std::cout << "ffmpeg -framerate 24 -i frame_%04d.bmp -c:v libx264 -pix_fmt yuv420p output_video.mp4" << std::endl;

    return 0;
}