#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <sys/stat.h> // For mkdir (POSIX)
#include <sys/types.h> // For mkdir (POSIX)

#include "file_io.h"

const float PI = 3.14159265358979f;

struct Vec3f {
    float x, y, z;

    Vec3f(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}

    Vec3f operator+(const Vec3f& other) const { return Vec3f(x + other.x, y + other.y, z + other.z); }
    Vec3f operator-(const Vec3f& other) const { return Vec3f(x - other.x, y - other.y, z - other.z); }
    Vec3f operator*(float scalar) const { return Vec3f(x * scalar, y * scalar, z * scalar); }
    float dot(const Vec3f& other) const { return x * other.x + y * other.y + z * other.z; }
    Vec3f cross(const Vec3f& other) const {
        return Vec3f(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    Vec3f normalize() const {
        float l = length();
        if (l == 0.0f) return Vec3f(0.0f, 0.0f, 0.0f); // Or handle error
        return Vec3f(x / l, y / l, z / l);
    }
};

// Overload the << operator for Vec3f for easy printing
std::ostream& operator<<(std::ostream& os, const Vec3f& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

// Black Hole Parameters & Ray Tracing Constants
const Vec3f BH_POSITION(50.0f, 0.0f, 0.0f); // Example: In front of camera if camera looks along +X
const float BH_RSCHWARZSCHILD_RADIUS = 2.0f; // Schwarzschild radius. Adjust for visual effect.
const float EVENT_HORIZON_RADIUS = BH_RSCHWARZSCHILD_RADIUS;
// const float PHOTON_SPHERE_RADIUS = 1.5f * BH_RSCHWARZSCHILD_RADIUS; // For reference

const float INTEGRATION_DISTANCE_MULTIPLIER = 50.0f; // How far from BH (in rs units) integration starts/ends
const int NUM_INTEGRATION_STEPS = 200;       // Number of steps for Euler integration
const float EPSILON_BH = 1e-4f;                   // Small number to avoid division by zero near BH center

// struct RayTraceResult MUST be defined after Vec3f but before its use
struct RayTraceResult {
    Vec3f finalDir;       // The final direction of the ray after passing the black hole
    bool hitEventHorizon; // True if the ray entered the event horizon

    RayTraceResult() : finalDir(0,0,0), hitEventHorizon(false) {}
};

struct Camera {
    Vec3f position;
    Vec3f lookAt;
    Vec3f worldUp;
    float fovY_degrees; // Vertical field of view in degrees

    Camera(Vec3f pos, Vec3f target, Vec3f up, float fov)
        : position(pos), lookAt(target), worldUp(up), fovY_degrees(fov) {}
};

// Calculates the pixel value from an equirectangular image in the direction of a 3D vector.
// imageData: flat vector of RGB pixels.
// imageWidth, imageHeight: dimensions of the equirectangular image.
// direction: 3D direction vector normalized to length 1.
// Returns a vector of 3 unsigned chars: [R, G, B].
std::vector<unsigned char> getPixelFromDirection(
    const std::vector<unsigned char>& imageData,
    int imageWidth,
    int imageHeight,
    const Vec3f& direction)
{


    // Convert Cartesian to Spherical coordinates
    // phi (azimuth/longitude): atan2(y, x). Range: [-PI, PI]
    float phi = std::atan2(direction.y, direction.x);
    // theta (polar/colatitude): acos(z). Range: [0, PI]
    // Clamp normZ to [-1, 1] to avoid domain errors with acos due to potential floating point inaccuracies
    float clamped_normZ = std::max(-1.0f, std::min(1.0f, direction.z));
    float theta = std::acos(clamped_normZ);

    // Map spherical coordinates to normalized texture coordinates (u, v)
    // u maps phi from [-PI, PI] to [0, 1]
    float u = (phi + static_cast<float>(PI)) / (2.0f * static_cast<float>(PI));
    // v maps theta from [0, PI] to [0, 1]
    float v = theta / static_cast<float>(PI);

    // Scale normalized coordinates to pixel coordinates using rounding
    int px = static_cast<int>(std::round(u * imageWidth));
    int py = static_cast<int>(std::round(v * imageHeight));

    // Clamp pixel coordinates to be within image bounds
    // For u=1 (phi=PI), px could be imageWidth. For v=1 (theta=PI), py could be imageHeight.
    if (px >= imageWidth) px = imageWidth - 1;
    if (py >= imageHeight) py = imageHeight - 1;
    // Ensure px, py are not negative (shouldn't happen with u,v in [0,1] but good for safety)
    if (px < 0) px = 0;
    if (py < 0) py = 0;

    // Calculate 1D index for the pixel in the imageData array (RGB format)
    // The BMP reading function already handles vertical flipping if needed,
    // so imageData[0] is top-left.
    // v=0 (North Pole) maps to py=0 (top row).
    // v=1 (South Pole) maps to py=imageHeight-1 (bottom row).
    int pixel_idx = (py * imageWidth + px) * 3;

    // Basic safety check for index bounds, though clamping px,py should prevent out-of-bounds
    // if imageData has the correct size (imageWidth * imageHeight * 3).
    if (pixel_idx < 0 || pixel_idx + 2 >= imageData.size()) {
        std::cerr << "Error: Calculated pixel index (" << pixel_idx << ") is out of bounds for image size " << imageData.size() << "." << std::endl;
        std::cerr << "  u: " << u << ", v: " << v << ", px: " << px << ", py: " << py << std::endl;
        std::cerr << "  imageWidth: " << imageWidth << ", imageHeight: " << imageHeight << std::endl;
        // Return black or throw an exception for an error
        return {0, 0, 0}; 
    }

    // Retrieve RGB values
    unsigned char r = imageData[pixel_idx + 0];
    unsigned char g = imageData[pixel_idx + 1];
    unsigned char b = imageData[pixel_idx + 2];

    return {r, g, b};
}

// Traces a ray near a black hole using 2D numerical integration.
// rayOrigin: The starting point of the ray (e.g., camera position, but effectively infinity for this model).
// rayDir: The initial direction of the ray (normalized).
// Returns RayTraceResult containing the final direction and event horizon hit status.
RayTraceResult traceRayNearBlackHole(const Vec3f& rayOrigin, const Vec3f& rayDir) {
    RayTraceResult result;
    result.finalDir = rayDir; // Default to original direction if no interaction
    result.hitEventHorizon = false;

    // Vector from ray origin to black hole position
    Vec3f L = BH_POSITION - rayOrigin;
    
    // Distance along the ray to the point of closest approach to BH_POSITION (if ray continues straight)
    float t_ca = L.dot(rayDir);

    // If t_ca < 0, black hole is behind the ray's starting point and ray is moving away.
    // We could optimize by returning if it's far behind and moving away, but for simplicity now, we proceed.
    // However, a very distant BH might not need integration.

    // 3D position of closest approach of the undeflected ray to the black hole
    Vec3f P_ca = rayOrigin + rayDir * t_ca;
    
    // 3D impact parameter vector (from BH to P_ca)
    Vec3f b_vec_3d = P_ca - BH_POSITION;
    float b_impact_param_3d = b_vec_3d.length();

    // If impact parameter is huge, deflection is negligible.
    // Let's define a max interaction distance based on Schwarzschild radius.
    // If b_impact_param_3d is much larger than INTEGRATION_DISTANCE_MULTIPLIER * BH_RSCHWARZSCHILD_RADIUS,
    // we can assume no deflection. For now, we integrate all rays that might come close.
    
    // Optimization: If the straight path of the ray intersects the event horizon sphere,
    // consider it absorbed. This is a 3D check.
    // Intersection of a ray (O + tD) with a sphere (center C, radius R_eh):
    // (D.D)t^2 + 2D.(O-C)t + (O-C).(O-C) - R_eh^2 = 0
    // Here, O=rayOrigin, D=rayDir, C=BH_POSITION, R_eh=EVENT_HORIZON_RADIUS
    Vec3f O_minus_C = rayOrigin - BH_POSITION;
    float a_quad = rayDir.dot(rayDir); // Should be 1.0 if rayDir is normalized
    float b_quad = 2.0f * rayDir.dot(O_minus_C);
    float c_quad = O_minus_C.dot(O_minus_C) - EVENT_HORIZON_RADIUS * EVENT_HORIZON_RADIUS;
    float discriminant = b_quad * b_quad - 4.0f * a_quad * c_quad;

    if (discriminant >= 0.0f) {
        float t0 = (-b_quad - std::sqrt(discriminant)) / (2.0f * a_quad);
        float t1 = (-b_quad + std::sqrt(discriminant)) / (2.0f * a_quad);
        if (t0 > 0 || t1 > 0) { // Intersection point is in front of the ray
            // Check if closest intersection is positive
            float t_intersect = (t0 > 0 && t1 > 0) ? std::min(t0,t1) : std::max(t0,t1);
            if (t_intersect > 0) {
                 // Potentially more robust: check if the segment [origin, origin + t_intersect*dir] is close enough to BH
                 // For now, if the infinite ray intersects, and the intersection point is in front and within a certain range
                 // or if the closest approach point t_ca is positive and b_impact_param_3d is less than EH.
                if (b_impact_param_3d < EVENT_HORIZON_RADIUS && t_ca > 0) {
                    result.hitEventHorizon = true;
                    return result;
                }
            }
        }
    }
    
    // Define the 2D integration plane.
    // The "x" axis of this 2D plane will be aligned with the original rayDir.
    // The "y" axis of this 2D plane will be in the direction of the 3D impact parameter.
    // The origin of this 2D plane is where the ray *would* pass the black hole at its closest approach if undeflected.
    // Let s be the coordinate along the original ray direction.
    // The black hole is at (s=0, y_2d = b_impact_param_3d) in this conceptual setup if ray comes from -s.

    // Integration occurs in a 2D plane. The light ray is assumed to travel mostly along one axis (say, 2D x-axis)
    // and the black hole is offset from this axis by the 2D impact parameter b_2d.
    // The provided equations are for x and y in the plane of deflection.

    // We need to define the 2D plane. Let the original ray direction be `s_dir_3d = rayDir`.
    // The `y_dir_2d_3d` (y-axis of 2D plane in 3D space) should be perpendicular to `s_dir_3d` and point towards the BH from the ray path.
    // If b_impact_param_3d is very small, b_vec_3d is ill-defined or zero. Handle this (ray goes through center).
    Vec3f y_axis_2d_plane; // This is the effective "y" direction in 3D for the 2D plane
    if (b_impact_param_3d < EPSILON_BH) { 
        // Ray is heading directly at or very near the center of the black hole.
        // Any such ray should hit the event horizon if t_ca > 0.
        // The previous 3D check should ideally catch this if EVENT_HORIZON_RADIUS > 0.
        // If it reaches here, it means it passes *extremely* close but didn't trigger the 3D EH hit.
        // This can happen if EH_RADIUS is tiny or zero. For safety, mark as hit.
        if (t_ca > 0) { // And BH is in front
             result.hitEventHorizon = true;
             return result;
        }
        // If BH is behind, or ray doesn't go towards it, just let it pass undeflected.
        return result; // No deflection, no EH hit
    }
    y_axis_2d_plane = b_vec_3d.normalize(); // Direction from ray path to BH at closest approach, in 3D

    // The x-axis of the 2D integration plane (in 3D space) is the original ray direction.
    Vec3f x_axis_2d_plane = rayDir; 

    // The z-axis of this 2D integration plane (normal to the plane) -- not strictly needed for 2D var updates
    // Vec3f z_axis_2d_plane = x_axis_2d_plane.cross(y_axis_2d_plane).normalize();

    float s_max_integration_dist = INTEGRATION_DISTANCE_MULTIPLIER * BH_RSCHWARZSCHILD_RADIUS;
    if (s_max_integration_dist < b_impact_param_3d * 2.5f) { // Ensure integration starts/ends far enough if impact param is large
        s_max_integration_dist = b_impact_param_3d * 2.5f;
    }
    if (s_max_integration_dist < 20.0f * BH_RSCHWARZSCHILD_RADIUS) { // Minimum integration distance, esp. if EH is large or b is small
        s_max_integration_dist = 20.0f * BH_RSCHWARZSCHILD_RADIUS;
    }
    if (s_max_integration_dist <= 0.0f) s_max_integration_dist = 10.0f; // Fallback if rs is zero or very small

    // Initial 2D state variables for the ODEs:
    float x_pos = -s_max_integration_dist;    // Ray starts far to the "left" of BH (BH at origin of this 2D system)
    float y_pos = b_impact_param_3d;          // Offset by the impact parameter

    float r_init = std::sqrt(x_pos * x_pos + y_pos * y_pos);
    if (r_init <= EVENT_HORIZON_RADIUS) { // Should not happen if s_max_integration_dist is large enough
        result.hitEventHorizon = true;
        return result;
    }
    float term_rs_over_r_init = BH_RSCHWARZSCHILD_RADIUS / r_init;
    if (std::abs(1.0f - term_rs_over_r_init) < EPSILON_BH || (1.0f - term_rs_over_r_init) <= 0.0f) {
         result.hitEventHorizon = true; // n is undefined or infinite at start, likely too close or error
         return result;
    }
    float n_init = 1.0f / (1.0f - term_rs_over_r_init);

    float px_val = n_init * 1.0f; // Initial direction (1,0) in 2D plane, so p_x = n*1
    float py_val = n_init * 0.0f; // p_y = n*0

    float ds_step = (2.0f * s_max_integration_dist) / static_cast<float>(NUM_INTEGRATION_STEPS);

    for (int i = 0; i < NUM_INTEGRATION_STEPS; ++i) {
        float r_sq = x_pos * x_pos + y_pos * y_pos;
        float r_current = std::sqrt(r_sq);

        if (r_current <= EVENT_HORIZON_RADIUS) {
            result.hitEventHorizon = true;
            return result;
        }

        float term_rs_over_r_current = BH_RSCHWARZSCHILD_RADIUS / r_current;
        // Check for r_current being too close to rs or inside, making n undefined or problematic
        if (std::abs(1.0f - term_rs_over_r_current) < EPSILON_BH) { 
            // At or very near photon sphere, n approaches infinity. Ray path is highly unstable.
            // Depending on definition, could treat as absorbed or scattered unpredictably.
            result.hitEventHorizon = true; // Simplification: treat as absorbed / too unstable
            return result;
        }
        if ((1.0f - term_rs_over_r_current) <= 0.0f) { // Inside Schwarzschild radius, n is ill-defined in this formula for external space
            result.hitEventHorizon = true; // Should have been caught by r_current <= EVENT_HORIZON_RADIUS
            return result;
        }
        float n_current = 1.0f / (1.0f - term_rs_over_r_current);
        
        // Avoid division by zero for r_cubed if r_current is extremely small (though EH check should catch it)
        if (r_current < EPSILON_BH) { 
             result.hitEventHorizon = true; 
             return result;
        }
        float r_cubed = r_sq * r_current;

        // dp_x/ds = -n^2 * rs * x / r^3
        // dp_y/ds = -n^2 * rs * y / r^3
        float accel_common_factor = -n_current * n_current * BH_RSCHWARZSCHILD_RADIUS / r_cubed;
        float dp_x = accel_common_factor * x_pos * ds_step;
        float dp_y = accel_common_factor * y_pos * ds_step;

        // dx/ds = p_x / n
        // dy/ds = p_y / n
        // Note: n_current can be very large if r_current is close to rs.
        // If n_current is huge, dx/ds and dy/ds become small (speed of light slowdown near BH).
        float dx = (px_val / n_current) * ds_step;
        float dy = (py_val / n_current) * ds_step;

        px_val += dp_x;
        py_val += dp_y;
        
        x_pos += dx;
        y_pos += dy;

        // Early exit optimization
        if (i > NUM_INTEGRATION_STEPS / 2) { // Check only in the latter half of integration
            if (x_pos > EPSILON_BH) { // Ray is on the "outgoing" side (positive x_pos)
                bool is_moving_radially_outward = (x_pos * px_val + y_pos * py_val) > 0;
                // r_current was calculated at the beginning of this loop iteration
                if (is_moving_radially_outward && (r_current > s_max_integration_dist * 0.90f)) {
                    // std::cout << "Early exit at step " << i << " for ray with b=" << b_impact_param_3d << std::endl;
                    break; // Exit integration loop early
                }
            }
        }
    }

    // After integration, convert final optical momentum (px_val, py_val) to a 2D direction vector T = p/n.
    float r_final = std::sqrt(x_pos*x_pos + y_pos*y_pos); 
    if (r_final <= EVENT_HORIZON_RADIUS) { // Final check just in case last step entered EH
        result.hitEventHorizon = true;
        return result;
    }
    float term_rs_over_r_final = BH_RSCHWARZSCHILD_RADIUS / r_final;
    if (std::abs(1.0f - term_rs_over_r_final) < EPSILON_BH || (1.0f - term_rs_over_r_final) <= 0.0f) {
         result.hitEventHorizon = true; // n is undefined or infinite at end, consider ray lost
         return result;
    }
    float n_final = 1.0f / (1.0f - term_rs_over_r_final);

    float final_tx = px_val / n_final;
    float final_ty = py_val / n_final;

    // Normalize the final 2D direction vector due to potential accumulated numerical errors.
    float final_t_len = std::sqrt(final_tx * final_tx + final_ty * final_ty);
    if (final_t_len < EPSILON_BH) {
        // Direction is near zero, something went wrong. Return original (undeflected) direction.
        // Or, if this happens after strong interaction, it might be better to mark as absorbed.
        // For now, let's assume it implies an issue and return undeflected or last known good direction.
        result.finalDir = rayDir; // Fallback to initial rayDir
        return result; 
    }
    float final_vx_norm = final_tx / final_t_len;
    float final_vy_norm = final_ty / final_t_len;

    result.finalDir = (x_axis_2d_plane * final_vx_norm + y_axis_2d_plane * final_vy_norm).normalize();

    return result;
}

// Renders an image from the camera's perspective using the equirectangular background.
// camera: The camera object defining the viewpoint.
// backgroundData, backgroundWidth, backgroundHeight: The equirectangular image data and dimensions.
// outputWidth, outputHeight: Dimensions of the desired rendered image.
// Returns a flat vector of RGB pixels for the rendered image.
std::vector<unsigned char> renderPerspective(
    const Camera& camera,
    const std::vector<unsigned char>& backgroundData,
    int backgroundWidth,
    int backgroundHeight,
    int outputWidth,
    int outputHeight)
{
    std::vector<unsigned char> outputImage(outputWidth * outputHeight * 3);

    // Camera basis vectors
    Vec3f viewDir = (camera.lookAt - camera.position).normalize();
    Vec3f camRight = viewDir.cross(camera.worldUp).normalize();
    // Recompute camera up to ensure it's orthogonal to viewDir and camRight
    Vec3f camUp = camRight.cross(viewDir).normalize(); 

    float aspectRatio = static_cast<float>(outputWidth) / static_cast<float>(outputHeight);
    float fovY_radians = camera.fovY_degrees * (PI / 180.0f);
    
    // Height of the view plane at distance 1
    float viewPlaneHeight = 2.0f * std::tan(fovY_radians / 2.0f);
    float viewPlaneWidth = viewPlaneHeight * aspectRatio;

    for (int j = 0; j < outputHeight; ++j) {
        for (int i = 0; i < outputWidth; ++i) {
            float Px = (2.0f * (static_cast<float>(i) + 0.5f) / static_cast<float>(outputWidth) - 1.0f);
            float Py = (1.0f - 2.0f * (static_cast<float>(j) + 0.5f) / static_cast<float>(outputHeight));

            Vec3f initialRayDirection = (
                viewDir +
                camRight * Px * (viewPlaneWidth / 2.0f) +
                camUp * Py * (viewPlaneHeight / 2.0f)
            ).normalize();

            // Trace the ray near the black hole
            RayTraceResult traceResult = traceRayNearBlackHole(camera.position, initialRayDirection);

            int idx = (j * outputWidth + i) * 3;
            if (traceResult.hitEventHorizon) {
                outputImage[idx + 0] = 0; // R = Black
                outputImage[idx + 1] = 0; // G = Black
                outputImage[idx + 2] = 0; // B = Black
            } else {
                std::vector<unsigned char> pixelColor = getPixelFromDirection(
                    backgroundData, backgroundWidth, backgroundHeight,
                    traceResult.finalDir // Use the deflected direction
                );

                if (!pixelColor.empty()) {
                    outputImage[idx + 0] = pixelColor[0]; // R
                    outputImage[idx + 1] = pixelColor[1]; // G
                    outputImage[idx + 2] = pixelColor[2]; // B
                } else {
                    // Should not happen if getPixelFromDirection handles errors and returns black
                    outputImage[idx + 0] = 255; // Magenta for error indication
                    outputImage[idx + 1] = 0;
                    outputImage[idx + 2] = 255;
                }
            }
        }
    }
    return outputImage;
}

int main() {
    const char* frame_output_dir = "frames_temp";

    // Create the output directory if it doesn't exist (POSIX specific)
    // For Windows, you might need #include <direct.h> and _mkdir()
    // Or use C++17 <filesystem> if available and preferred.
    struct stat st = {0};
    if (stat(frame_output_dir, &st) == -1) { // Check if directory exists
        #if defined(_WIN32) || defined(_WIN64)
            // For Windows, you would use _mkdir(frame_output_dir) from <direct.h>
            // This example focuses on POSIX for now.
            std::cout << "Attempting to create directory (Windows placeholder): " << frame_output_dir << std::endl;
            // if (_mkdir(frame_output_dir) != 0) { // Placeholder, requires #include <direct.h>
            //     std::cerr << "Error: Could not create directory " << frame_output_dir << std::endl;
            //     return 1; // Exit if directory creation fails
            // }
        #else 
            // POSIX mkdir (Linux, macOS)
            std::cout << "Creating directory: " << frame_output_dir << std::endl;
            if (mkdir(frame_output_dir, 0775) != 0 && errno != EEXIST) { // 0775 gives rwx for owner/group, r-x for others
                std::cerr << "Error: Could not create directory " << frame_output_dir << ". Errno: " << errno << std::endl;
                perror("mkdir error"); // More detailed error
                return 1; // Exit if directory creation fails
            }
        #endif
    }

    int bg_width, bg_height;
    std::vector<unsigned char> backgroundImageData = readBMP("../assets/galaxy.bmp", bg_width, bg_height);

    if (backgroundImageData.empty()) {
        std::cerr << "Failed to load background image. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Successfully read background BMP image with dimensions: " << bg_width << "x" << bg_height << std::endl;

    // Animation Parameters
    const int num_frames = 120; // Number of frames for the animation (e.g., 5 seconds at 24 fps)
    const float orbit_radius = 60.0f; // Radius of the camera's orbit around BH_POSITION
    // const float camera_z_offset = BH_POSITION.z + 20.0f; // Optional: elevate camera slightly
    const float camera_z_offset = BH_POSITION.z; // Keep camera in the same z-plane as BH for simplicity

    // Camera parameters that change per frame are position and lookAt (always BH_POSITION)
    // worldUp and fovY can remain constant for this animation
    Vec3f worldUpVector(0.0f, 0.0f, 1.0f);     // Z is up for the camera
    float fieldOfViewY = 75.0f;                // Wider FOV might be nice
    
    // Define output image dimensions (user already set these)
    int outputWidth = 720;
    int outputHeight = 480;

    char filename_buffer[256]; // For formatting output filenames

    std::cout << "Starting animation rendering: " << num_frames << " frames." << std::endl;
    std::cout << "Frames will be saved in: " << frame_output_dir << "/" << std::endl;

    for (int frame = 0; frame < num_frames; ++frame) {
        float angle = PI/2 + PI * static_cast<float>(frame) / static_cast<float>(num_frames);

        // Camera orbits in the XY plane around BH_POSITION
        Vec3f cameraPosition(
            BH_POSITION.x + orbit_radius * std::cos(angle),
            BH_POSITION.y + orbit_radius * std::sin(angle),
            camera_z_offset 
        );
        
        // Camera always looks at the black hole
        Vec3f lookAtTarget = BH_POSITION;

        Camera camera(cameraPosition, lookAtTarget, worldUpVector, fieldOfViewY);

        snprintf(filename_buffer, sizeof(filename_buffer), "%s/frame_%04d.bmp", frame_output_dir, frame);
        std::cout << "Rendering frame " << (frame + 1) << "/" << num_frames << " to " << filename_buffer << "..." << std::endl;

        std::vector<unsigned char> renderedImage = renderPerspective(
            camera,
            backgroundImageData,
            bg_width,
            bg_height,
            outputWidth,
            outputHeight
        );
    
        if (!renderedImage.empty()) {
            saveBMP(filename_buffer, renderedImage, outputWidth, outputHeight);
        } else {
            std::cerr << "Error: renderPerspective returned an empty image for frame " << frame << std::endl;
            // Optionally create a dummy/error frame or skip
        }
    }

    std::cout << "Animation rendering finished." << std::endl;
    std::cout << "You can now use a tool like ffmpeg to create a video from the frame_xxxx.bmp files." << std::endl;
    std::cout << "Example ffmpeg command:" << std::endl;
    std::cout << "ffmpeg -framerate 24 -i frame_%04d.bmp -c:v libx264 -pix_fmt yuv420p output_video.mp4" << std::endl;
    
    return 0;
}