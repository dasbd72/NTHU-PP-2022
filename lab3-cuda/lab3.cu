#include <png.h>
#include <zlib.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8
#define START_X (MASK_X / 2)
#define START_Y (MASK_Y / 2)
#define BLOCK_X 16
#define BLOCK_Y 8
#define FLOAT_SOBEL 0

#ifdef DEBUG
#include <chrono>
#define DEBUG_PRINT(fmt, args...) printf(fmt, ##args);
#define CUDAEXE(func)                                            \
    do {                                                         \
        cudaError_t err = func;                                  \
        if (err != cudaSuccess)                                  \
            DEBUG_PRINT("Error: %s\n", cudaGetErrorString(err)); \
    } while (false);
#else
#define DEBUG_PRINT(fmt, args...)
#define CUDAEXE(func) func;
#endif

// clang-format off
__constant__ short mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};
// clang-format on

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels, unsigned* height_pad, unsigned* width_pad) {
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);
    *height_pad = (unsigned int)ceil((float)(*height) / BLOCK_Y) * BLOCK_Y;
    *width_pad = (unsigned int)ceil((float)(*width) / BLOCK_X) * BLOCK_X;
    size_t totsize = *channels * (*width_pad + MASK_X - 1) * (*height_pad + MASK_Y - 1) * sizeof(unsigned char);

    if ((*image = (unsigned char*)malloc(totsize)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }
    memset(*image, 0, totsize);

    for (i = 0; i < *height; ++i)
        row_pointers[i] = *image + (i + START_Y) * *channels * (*width_pad + MASK_X - 1) + (START_X * *channels);
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels, const unsigned height_pad, const unsigned width_pad) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + (i + START_Y) * (width_pad + MASK_X - 1) * channels * sizeof(unsigned char) + (START_X * channels);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels, unsigned heigh_pad, unsigned width_pad) {
    int x, y, i, v, u;
    int R, G, B;
    double val[MASK_N * 3] = {0.0};

    x = blockIdx.x * blockDim.x + threadIdx.x + START_X;
    y = blockIdx.y * blockDim.y + threadIdx.y + START_Y;
#pragma unroll 2
    for (i = 0; i < MASK_N; ++i) {
        val[i * 3 + 2] = 0.0;
        val[i * 3 + 1] = 0.0;
        val[i * 3] = 0.0;

#pragma unroll 5
        for (v = -2; v < 3; ++v) {
#pragma unroll 5
            for (u = -2; u < 3; ++u) {
                R = s[channels * ((width_pad + 4) * (y + v) + (x + u)) + 2];
                G = s[channels * ((width_pad + 4) * (y + v) + (x + u)) + 1];
                B = s[channels * ((width_pad + 4) * (y + v) + (x + u)) + 0];
                val[i * 3 + 2] += R * mask[i][u + 2][v + 2];
                val[i * 3 + 1] += G * mask[i][u + 2][v + 2];
                val[i * 3 + 0] += B * mask[i][u + 2][v + 2];
            }
        }
    }

#if FLOAT_SOBEL == 0
    double totalR = 0.0;
    double totalG = 0.0;
    double totalB = 0.0;
#else
    float totalR = 0.0;
    float totalG = 0.0;
    float totalB = 0.0;
#endif

#pragma unroll 2
    for (i = 0; i < MASK_N; ++i) {
        totalR += val[i * 3 + 2] * val[i * 3 + 2];
        totalG += val[i * 3 + 1] * val[i * 3 + 1];
        totalB += val[i * 3 + 0] * val[i * 3 + 0];
    }

#if FLOAT_SOBEL == 0
    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;
#else
    totalR = sqrtf(totalR) / SCALE;
    totalG = sqrtf(totalG) / SCALE;
    totalB = sqrtf(totalB) / SCALE;
#endif

    const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
    const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
    const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
    t[channels * ((width_pad + 4) * y + x) + 2] = cR;
    t[channels * ((width_pad + 4) * y + x) + 1] = cG;
    t[channels * ((width_pad + 4) * y + x) + 0] = cB;
}

int main(int argc, char** argv) {
    assert(argc == 3);
    int deviceId;
    cudaGetDevice(&deviceId);
    DEBUG_PRINT("deviceId: %d\n", deviceId);

    unsigned height, width, channels, height_pad, width_pad;
    unsigned char* host_s = NULL;
    unsigned char* host_t = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels, &height_pad, &width_pad);
    size_t totsize = channels * (width_pad + MASK_X - 1) * (height_pad + MASK_Y - 1) * sizeof(unsigned char);

    dim3 blk(BLOCK_X, BLOCK_Y);
    dim3 grid(width_pad / BLOCK_X, height_pad / BLOCK_Y);
    unsigned char* dev_s = NULL;
    unsigned char* dev_t = NULL;

    /* Profile */
    DEBUG_PRINT("%u %u %u %u\n", height, width, height_pad, width_pad);
    DEBUG_PRINT("%u %u %lu\n", height_pad + 4, width_pad + 4, totsize);

    /* Allocate device memory */
    CUDAEXE(cudaMalloc(&dev_s, totsize));
    CUDAEXE(cudaMalloc(&dev_t, totsize));

    /* Copy host to device */
    CUDAEXE(cudaMemcpy(dev_s, host_s, totsize, cudaMemcpyHostToDevice));
    CUDAEXE(cudaMemset(dev_t, 0, totsize));

    /* Sobel */
    sobel<<<grid, blk>>>(dev_s, dev_t, height, width, channels, height_pad, width_pad);

    /* Copy device to host */
    host_t = (unsigned char*)malloc(totsize);
    CUDAEXE(cudaMemcpyAsync(host_t, dev_t, totsize, cudaMemcpyDeviceToHost));
    CUDAEXE(cudaGetLastError());
    CUDAEXE(cudaDeviceSynchronize());

    /* Output */
    write_png(argv[2], host_t, height, width, channels, height_pad, width_pad);

    /* Finalize */
    free(host_s);
    free(host_t);
    cudaFree(dev_s);
    cudaFree(dev_t);
    return 0;
}
