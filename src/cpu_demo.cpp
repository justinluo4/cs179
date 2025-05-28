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

// Test function to read a BMP and save it back out, to verify read/write integrity
void testAndSaveOriginalBMP(const std::string& inputFilename, const std::string& outputFilename) {
    int width, height;
    std::cout << "Testing BMP read/write by copying " << inputFilename << " to " << outputFilename << std::endl;
    std::vector<unsigned char> imageData = readBMP(inputFilename.c_str(), width, height);

    if (imageData.empty()) {
        std::cerr << "Failed to read " << inputFilename << " for copy test." << std::endl;
        return;
    }

    std::cout << "Read " << inputFilename << " for copy test: " << width << "x" << height << std::endl;
    saveBMP(outputFilename, imageData, width, height);
    std::cout << "Copy test finished. Check " << outputFilename << std::endl;
}

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

    // The z-axis of this 2D integration plane (normal to the plane)
    Vec3f z_axis_2d_plane = x_axis_2d_plane.cross(y_axis_2d_plane).normalize();

    // Initial state for 2D integration:
    // The ray starts far away, say at s = -s_max_integration_dist, moving towards s = +s_max_integration_dist
    // The black hole is effectively at (x_2d=0, y_2d=0) in the coordinates of the ODEs.
    // So, the light ray starts at x_2d = some_large_negative_value, y_2d = b_impact_param_3d (the 2D impact parameter).
    float s_max_integration_dist = INTEGRATION_DISTANCE_MULTIPLIER * BH_RSCHWARZSCHILD_RADIUS;
    if (s_max_integration_dist < b_impact_param_3d * 2.0f) { // Ensure integration starts far enough if impact param is large
        s_max_integration_dist = b_impact_param_3d * 2.0f;
    }
    if (s_max_integration_dist < 10.0f * BH_RSCHWARZSCHILD_RADIUS) { // Minimum integration distance if EH is large
        s_max_integration_dist = 10.0f * BH_RSCHWARZSCHILD_RADIUS;
    }


    // 2D state variables for the ODEs: (x, y) position, (vx, vy) "velocity" (direction components)
    float x = -s_max_integration_dist;         // Ray starts far to the "left" of BH
    float y = b_impact_param_3d;             // Offset by the impact parameter
    float vx = 1.0f;                         // Initially moving purely in +x direction (normalized speed)
    float vy = 0.0f;

    float ds_step = (2.0f * s_max_integration_dist) / static_cast<float>(NUM_INTEGRATION_STEPS);

    for (int i = 0; i < NUM_INTEGRATION_STEPS; ++i) {
        float r_sq = x * x + y * y;
        float r = std::sqrt(r_sq);

        if (r < EVENT_HORIZON_RADIUS) {
            result.hitEventHorizon = true;
            return result;
        }

        // Avoid division by zero if r is extremely small (though EH check should catch it)
        float r_cubed = r_sq * r;
        if (r_cubed < EPSILON_BH * EPSILON_BH * EPSILON_BH) { // Avoid div by zero if somehow past EH
             // This case should ideally not be reached if EH check is robust.
             // If it is, the ray is extremely close to the singularity; treat as absorbed or highly chaotic.
             result.hitEventHorizon = true; // Or just return undeflected if that makes more sense for an edge case
             return result;
        }

        // Equations from user: dvx/ds = -rs * x / r^3, dvy/ds = -rs * y / r^3
        // (Assuming n=1, c=1 in these units, so 2GM = rs)
        float common_factor = -BH_RSCHWARZSCHILD_RADIUS / r_cubed;
        
        float dvx = common_factor * x * ds_step;
        float dvy = common_factor * y * ds_step;

        vx += dvx;
        vy += dvy;
        
        // Normalize velocity vector to maintain it as a direction (optional but good practice for some integrators)
        // For simple Euler, it might accumulate error, but let's try without first.
        // float v_len = std::sqrt(vx*vx + vy*vy);
        // if (v_len > EPSILON_BH) { vx /= v_len; vy /= v_len; }

        x += vx * ds_step;
        y += vy * ds_step;
    }

    // After integration, (vx, vy) is the final 2D direction in the integration plane.
    // We need to convert this 2D direction back to a 3D world direction.
    // The original ray direction `x_axis_2d_plane` corresponds to (1,0) in 2D before deflection.
    // The `y_axis_2d_plane` corresponds to (0,1) in 2D before deflection.
    // The final 3D direction is a linear combination of these axes based on the final (vx, vy).
    // However, the (vx, vy) are components of the *deflected path* in the 2D plane coordinates where the ray started along x.
    // The final 2D direction (vx, vy) needs to be normalized.
    float final_v_len = std::sqrt(vx * vx + vy * vy);
    if (final_v_len < EPSILON_BH) {
        // Velocity is near zero, something went wrong or ray stopped. Return original.
        return result; 
    }
    float final_vx_norm = vx / final_v_len;
    float final_vy_norm = vy / final_v_len;

    // Reconstruct the 3D direction:
    // The ray initially traveled along x_axis_2d_plane.
    // Its final direction has a component final_vx_norm along the original x_axis_2d_plane
    // and a component final_vy_norm along the y_axis_2d_plane.
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
    // Test the BMP read/write functions directly
    testAndSaveOriginalBMP("../assets/galaxy.bmp", "background_direct_copy.bmp");

    // Original rendering code - you can comment this out or run it after the test
    int width, height;
    std::vector<unsigned char> imageData = readBMP("../assets/galaxy.bmp", width, height);
    
    std::cout << "Successfully read BMP image with dimensions: " << width << "x" << height << std::endl;
    std::cout << "Total pixels: " << width * height << std::endl;
    std::cout << "Image data size: " << imageData.size() << " bytes" << std::endl;

    // Define camera parameters
    Vec3f cameraPosition(0.0f, 0.0f, 0.0f);    // Camera at origin
    Vec3f lookAtTarget(1.0f, 0.0f, 0.0f);   // Looking down the -Z axis
    Vec3f worldUpVector(0.0f, 0.0f, 1.0f);     // Y is up
    float fieldOfViewY = 100.0f;                // 60 degrees vertical FOV

    Camera camera(cameraPosition, lookAtTarget, worldUpVector, fieldOfViewY);

    // Define output image dimensions
    int outputWidth = 1280;
    int outputHeight = 720;

    std::cout << "Rendering perspective image..." << std::endl;
    std::vector<unsigned char> renderedImage = renderPerspective(
        camera,
        imageData,     // Background (equirectangular) image data
        width,         // Background image width
        height,        // Background image height
        outputWidth,
        outputHeight
    );
    

    // Save the rendered image
    saveBMP("rendered_galaxy2.bmp", renderedImage, outputWidth, outputHeight);
    
    return 0;
}