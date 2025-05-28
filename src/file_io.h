#ifndef FILE_IO_H_
#define FILE_IO_H_

#include <vector>
#include <string>
#include <cstdint>

// BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t signature;      // "BM"
    uint32_t fileSize;       // Size of the BMP file in bytes
    uint16_t reserved1;      // Reserved
    uint16_t reserved2;      // Reserved
    uint32_t dataOffset;     // Offset to image data in bytes
    uint32_t headerSize;     // Header size in bytes
    int32_t width;          // Width of the image
    int32_t height;         // Height of the image
    uint16_t planes;         // Number of color planes
    uint16_t bitsPerPixel;   // Bits per pixel
    uint32_t compression;    // Compression type
    uint32_t imageSize;      // Image size in bytes
    int32_t xPixelsPerM;    // Pixels per meter in X
    int32_t yPixelsPerM;    // Pixels per meter in Y
    uint32_t colorsUsed;     // Number of colors used
    uint32_t colorsImportant;// Number of important colors
};
#pragma pack(pop)

// Function to read BMP file into an array
std::vector<unsigned char> readBMP(const char* filename, int& width, int& height);

// Utility function to save an image buffer (RGB) to a PPM file
void savePPM(const std::string& filename, const std::vector<unsigned char>& imageBuffer, int width, int height);

// Utility function to save an image buffer (RGB) to a BMP file
void saveBMP(const std::string& filename, const std::vector<unsigned char>& imageBuffer, int width, int height);

#endif // FILE_IO_H_ 