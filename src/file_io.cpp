#include "file_io.h"

#include <cstdio>   // For FILE, fopen, fread, fseek, fclose
#include <cstdlib>  // For exit, abs
#include <cstring>  // For std::memset
#include <fstream>  // For std::ofstream
#include <iostream> // For std::cerr, std::cout
#include <vector>   // For std::vector, explicitly include if file_io.h didn't bring it for some reason
#include <cmath>    // For std::abs

// Function to read BMP file into an array
std::vector<unsigned char> readBMP(const char* filename, int& width, int& height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {}; // Return empty vector on error
    }

    BMPHeader header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        std::cerr << "Error: Could not read BMP header from " << filename << std::endl;
        fclose(file);
        return {};
    }

    // Verify that it's a BMP file signature
    if (header.signature != 0x4D42) { // "BM" in hex
        std::cerr << "Error: Not a BMP file (invalid signature) in " << filename << ". Signature: " << std::hex << header.signature << std::dec << std::endl;
        fclose(file);
        return {};
    }

    // Validate critical BMP properties for this reader
    if (header.bitsPerPixel != 24) {
        std::cerr << "Error: BMP file \"" << filename << "\" is not 24 bits per pixel. Detected bpp: " << header.bitsPerPixel << std::endl;
        std::cerr << "  Header details - fileSize: " << header.fileSize << ", dataOffset: " << header.dataOffset << ", width: " << header.width << ", height: " << header.height << ", compression: " << header.compression << std::endl;
        fclose(file);
        return {};
    }
    if (header.compression != 0) {
        std::cerr << "Error: BMP file \"" << filename << "\" is compressed (compression type: " << header.compression << "). Only uncompressed BMPs are supported by this reader." << std::endl;
        std::cerr << "  Header details - fileSize: " << header.fileSize << ", dataOffset: " << header.dataOffset << ", width: " << header.width << ", height: " << header.height << ", bpp: " << header.bitsPerPixel << std::endl;
        fclose(file);
        return {};
    }
    if (header.width <= 0 || header.height == 0) {
         std::cerr << "Error: BMP file \"" << filename << "\" has invalid dimensions (width: " << header.width << ", height: " << header.height << ")." << std::endl;
        fclose(file);
        return {};
    }


    // Store dimensions
    width = header.width;
    height = std::abs(header.height); // Actual image height is absolute

    // Calculate row padding (each scanline is padded to a multiple of 4 bytes)
    int padding = (4 - (width * 3) % 4) % 4;
    
    std::vector<unsigned char> imageData(width * height * 3);
    
    // Seek to the beginning of the pixel data
    if (fseek(file, header.dataOffset, SEEK_SET) != 0) {
        std::cerr << "Error: Failed to seek to pixel data in " << filename << std::endl;
        fclose(file);
        return {};
    }

    unsigned char pixel_bgr[3]; // Temporary buffer to read BGR pixel

    for (int y_coord = 0; y_coord < height; y_coord++) {
        for (int x_coord = 0; x_coord < width; x_coord++) {
            int index;
            // BMP files can be stored top-down or bottom-up. header.height sign indicates this.
            // We always store it top-down in imageData vector.
            if (header.height > 0) { // Positive height: BMP is bottom-up, so store inversely in our top-down buffer
                 index = (height - 1 - y_coord) * width * 3 + x_coord * 3; 
            } else { // Negative height: BMP is top-down, store directly
                 index = y_coord * width * 3 + x_coord * 3;
            }

            if (fread(pixel_bgr, 1, 3, file) != 3) {
                std::cerr << "Error: Failed to read 3 bytes for pixel (" << x_coord << "," << y_coord << ") from " << filename << std::endl;
                fclose(file);
                return {}; 
            }
            imageData[index + 0] = pixel_bgr[2]; // File R -> imageData R
            imageData[index + 1] = pixel_bgr[1]; // File G -> imageData G
            imageData[index + 2] = pixel_bgr[0]; // File B -> imageData B
        }
        // After reading all pixels in a row, skip padding bytes
        if (padding > 0) {
            if (fseek(file, padding, SEEK_CUR) != 0) {
                 std::cerr << "Error: Failed to seek past padding for row " << y_coord << " in " << filename << std::endl;
                fclose(file);
                return {};
            }
        }
    }

    fclose(file);
    return imageData;
}

// Utility function to save an image buffer (RGB) to a PPM file
void savePPM(const std::string& filename, const std::vector<unsigned char>& imageBuffer, int width, int height) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    ofs << "P6\n" << width << " " << height << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(imageBuffer.data()), imageBuffer.size());
    ofs.close();
    std::cout << "Saved rendered image to " << filename << std::endl;
}

// Utility function to save an image buffer (RGB) to a BMP file
void saveBMP(const std::string& filename, const std::vector<unsigned char>& imageBuffer, int width, int height) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    BMPHeader header;
    std::memset(&header, 0, sizeof(header));

    int row_padded = (width * 3 + 3) & (~3);
    uint32_t imageSize = row_padded * height;
    uint32_t dataOffset = sizeof(BMPHeader);
    uint32_t fileSize = dataOffset + imageSize;

    header.signature = 0x4D42;
    header.fileSize = fileSize;
    header.dataOffset = dataOffset;
    header.headerSize = 40;
    header.width = width;
    header.height = height; // Positive for bottom-up rows in output
    header.planes = 1;
    header.bitsPerPixel = 24;
    header.compression = 0;
    header.imageSize = imageSize;
    header.xPixelsPerM = 0; 
    header.yPixelsPerM = 0;
    header.colorsUsed = 0;
    header.colorsImportant = 0;

    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));

    int padding = row_padded - (width * 3);
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            int bufferIdx = (y * width + x) * 3;
            unsigned char r = imageBuffer[bufferIdx + 0];
            unsigned char g = imageBuffer[bufferIdx + 1];
            unsigned char b = imageBuffer[bufferIdx + 2];
            
            ofs.write(reinterpret_cast<const char*>(&b), 1);
            ofs.write(reinterpret_cast<const char*>(&g), 1);
            ofs.write(reinterpret_cast<const char*>(&r), 1);
        }
        for (int k = 0; k < padding; ++k) {
            char padByte = 0;
            ofs.write(&padByte, 1);
        }
    }
    ofs.close();
    std::cout << "Saved rendered image to " << filename << std::endl;
} 