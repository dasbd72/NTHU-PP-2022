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
#define BLOCK_X 32
#define BLOCK_Y 16
#define FLOAT_SOBEL 1

#ifdef DEBUG
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

#ifdef TIMING
#include <ctime>
#define TIMING_START(arg)          \
    struct timespec __start_##arg; \
    clock_gettime(CLOCK_MONOTONIC, &__start_##arg);
#define TIMING_END(arg)                                                                       \
    {                                                                                         \
        struct timespec __temp_##arg, __end_##arg;                                            \
        double __duration_##arg;                                                              \
        clock_gettime(CLOCK_MONOTONIC, &__end_##arg);                                         \
        if ((__end_##arg.tv_nsec - __start_##arg.tv_nsec) < 0) {                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec - 1;              \
            __temp_##arg.tv_nsec = 1000000000 + __end_##arg.tv_nsec - __start_##arg.tv_nsec;  \
        } else {                                                                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec;                  \
            __temp_##arg.tv_nsec = __end_##arg.tv_nsec - __start_##arg.tv_nsec;               \
        }                                                                                     \
        __duration_##arg = __temp_##arg.tv_sec + (double)__temp_##arg.tv_nsec / 1000000000.0; \
        printf("%s took %lfs.\n", #arg, __duration_##arg);                                    \
    }
#else
#define TIMING_START(arg)
#define TIMING_END(arg)
#endif  // TIMING

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
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + (i + START_Y) * (width_pad + MASK_X - 1) * channels * sizeof(unsigned char) + (START_X * channels);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/* __global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels, unsigned heigh_pad, unsigned width_pad) {
    __shared__ unsigned char S[3][5][BLOCK_Y + 4];
    int x, y, i, v, u;
    unsigned char R, G, B;
    short val[MASK_N * 3] = {0};

    x = blockIdx.x * BLOCK_X + START_X;
    y = blockIdx.y * blockDim.y + threadIdx.y + START_Y;

    for (v = 0; threadIdx.y + v < BLOCK_Y + 4; v += BLOCK_Y) {
#pragma unroll 5
        for (u = 0; u < 5; u++) {
            S[2][u][threadIdx.y + v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 2];
            S[1][u][threadIdx.y + v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 1];
            S[0][u][threadIdx.y + v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 0];
        }
    }
    __syncthreads();

    {
#pragma unroll 2
        for (i = 0; i < MASK_N; ++i) {
            val[i * 3 + 2] = 0;
            val[i * 3 + 1] = 0;
            val[i * 3] = 0;

#pragma unroll 5
            for (v = 0; v < 5; ++v) {
#pragma unroll 5
                for (u = 0; u < 5; ++u) {
                    R = S[2][u][threadIdx.y + v];
                    G = S[1][u][threadIdx.y + v];
                    B = S[0][u][threadIdx.y + v];
                    val[i * 3 + 2] += R * mask[i][u][v];
                    val[i * 3 + 1] += G * mask[i][u][v];
                    val[i * 3 + 0] += B * mask[i][u][v];
                }
            }
        }

        float totalR = 0.0;
        float totalG = 0.0;
        float totalB = 0.0;

#pragma unroll 2
        for (i = 0; i < MASK_N; ++i) {
            totalR += val[i * 3 + 2] * val[i * 3 + 2];
            totalG += val[i * 3 + 1] * val[i * 3 + 1];
            totalB += val[i * 3 + 0] * val[i * 3 + 0];
        }

        totalR = sqrtf(totalR) / SCALE;
        totalG = sqrtf(totalG) / SCALE;
        totalB = sqrtf(totalB) / SCALE;

        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
        t[channels * ((width_pad + 4) * y + x) + 2] = cR;
        t[channels * ((width_pad + 4) * y + x) + 1] = cG;
        t[channels * ((width_pad + 4) * y + x) + 0] = cB;
    }

    for (x++; x < (blockIdx.x + 1) * BLOCK_X + START_X; x++) {
        for (v = 0; threadIdx.y + v < BLOCK_Y + 4; v += BLOCK_Y) {
#pragma unroll 4
            for (u = 0; u < 4; u++) {
                S[2][u][threadIdx.y + v] = S[2][u + 1][threadIdx.y + v];
                S[1][u][threadIdx.y + v] = S[1][u + 1][threadIdx.y + v];
                S[0][u][threadIdx.y + v] = S[0][u + 1][threadIdx.y + v];
            }
            S[2][u][threadIdx.y + v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 2];
            S[1][u][threadIdx.y + v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 1];
            S[0][u][threadIdx.y + v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 0];
        }
        __syncthreads();

#pragma unroll 2
        for (i = 0; i < MASK_N; ++i) {
            val[i * 3 + 2] = 0;
            val[i * 3 + 1] = 0;
            val[i * 3] = 0;

#pragma unroll 5
            for (v = 0; v < 5; ++v) {
#pragma unroll 5
                for (u = 0; u < 5; ++u) {
                    R = S[2][u][threadIdx.y + v];
                    G = S[1][u][threadIdx.y + v];
                    B = S[0][u][threadIdx.y + v];
                    val[i * 3 + 2] += R * mask[i][u][v];
                    val[i * 3 + 1] += G * mask[i][u][v];
                    val[i * 3 + 0] += B * mask[i][u][v];
                }
            }
        }

        float totalR = 0.0;
        float totalG = 0.0;
        float totalB = 0.0;

#pragma unroll 2
        for (i = 0; i < MASK_N; ++i) {
            totalR += val[i * 3 + 2] * val[i * 3 + 2];
            totalG += val[i * 3 + 1] * val[i * 3 + 1];
            totalB += val[i * 3 + 0] * val[i * 3 + 0];
        }

        totalR = sqrtf(totalR) / SCALE;
        totalG = sqrtf(totalG) / SCALE;
        totalB = sqrtf(totalB) / SCALE;

        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
        t[channels * ((width_pad + 4) * y + x) + 2] = cR;
        t[channels * ((width_pad + 4) * y + x) + 1] = cG;
        t[channels * ((width_pad + 4) * y + x) + 0] = cB;
    }
} */

__global__ void sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels, unsigned heigh_pad, unsigned width_pad) {
    extern __shared__ unsigned char S[3][BLOCK_X + 4][5];
    int x, y, i, v, u;
    unsigned char R, G, B;
    short val[MASK_N * 3] = {0};

    x = blockIdx.x * blockDim.x + threadIdx.x + START_X;
    y = blockIdx.y * BLOCK_Y + START_Y;

#pragma unroll 5
    for (v = 0; v < 5; v++) {
        for (u = 0; threadIdx.x + u < BLOCK_X + 4; u += BLOCK_X) {
            S[2][threadIdx.x + u][v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 2];
            S[1][threadIdx.x + u][v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 1];
            S[0][threadIdx.x + u][v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 0];
        }
    }
    __syncthreads();

    {
#pragma unroll 2
        for (i = 0; i < MASK_N; ++i) {
            val[i * 3 + 2] = 0;
            val[i * 3 + 1] = 0;
            val[i * 3] = 0;

#pragma unroll 5
            for (v = 0; v < 5; ++v) {
#pragma unroll 5
                for (u = 0; u < 5; ++u) {
                    R = S[2][threadIdx.x + u][v];
                    G = S[1][threadIdx.x + u][v];
                    B = S[0][threadIdx.x + u][v];
                    val[i * 3 + 2] += R * mask[i][u][v];
                    val[i * 3 + 1] += G * mask[i][u][v];
                    val[i * 3 + 0] += B * mask[i][u][v];
                }
            }
        }

        float totalR = 0.0;
        float totalG = 0.0;
        float totalB = 0.0;

#pragma unroll 2
        for (i = 0; i < MASK_N; ++i) {
            totalR += val[i * 3 + 2] * val[i * 3 + 2];
            totalG += val[i * 3 + 1] * val[i * 3 + 1];
            totalB += val[i * 3 + 0] * val[i * 3 + 0];
        }

        totalR = sqrtf(totalR) / SCALE;
        totalG = sqrtf(totalG) / SCALE;
        totalB = sqrtf(totalB) / SCALE;

        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
        t[channels * ((width_pad + 4) * y + x) + 2] = cR;
        t[channels * ((width_pad + 4) * y + x) + 1] = cG;
        t[channels * ((width_pad + 4) * y + x) + 0] = cB;
    }

    for (y++; y < (blockIdx.y + 1) * BLOCK_Y + START_Y; y++) {
#pragma unroll 4
        for (v = 0; v < 4; v++) {
            for (u = 0; threadIdx.x + u < BLOCK_X + 4; u += BLOCK_X) {
                S[2][threadIdx.x + u][v] = S[2][threadIdx.x + u][v + 1];
                S[1][threadIdx.x + u][v] = S[1][threadIdx.x + u][v + 1];
                S[0][threadIdx.x + u][v] = S[0][threadIdx.x + u][v + 1];
            }
        }
        for (u = 0; threadIdx.x + u < BLOCK_X + 4; u += BLOCK_X) {
            S[2][threadIdx.x + u][v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 2];
            S[1][threadIdx.x + u][v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 1];
            S[0][threadIdx.x + u][v] = s[((width_pad + 4) * (y - 2 + v) + (x - 2 + u)) * 3 + 0];
        }
        __syncthreads();

#pragma unroll 2
        for (i = 0; i < MASK_N; ++i) {
            val[i * 3 + 2] = 0.0;
            val[i * 3 + 1] = 0.0;
            val[i * 3] = 0.0;

#pragma unroll 5
            for (v = 0; v < 5; ++v) {
#pragma unroll 5
                for (u = 0; u < 5; ++u) {
                    R = S[2][threadIdx.x + u][v];
                    G = S[1][threadIdx.x + u][v];
                    B = S[0][threadIdx.x + u][v];
                    val[i * 3 + 2] += R * mask[i][u][v];
                    val[i * 3 + 1] += G * mask[i][u][v];
                    val[i * 3 + 0] += B * mask[i][u][v];
                }
            }
        }

        float totalR = 0.0;
        float totalG = 0.0;
        float totalB = 0.0;

#pragma unroll 2
        for (i = 0; i < MASK_N; ++i) {
            totalR += val[i * 3 + 2] * val[i * 3 + 2];
            totalG += val[i * 3 + 1] * val[i * 3 + 1];
            totalB += val[i * 3 + 0] * val[i * 3 + 0];
        }

        totalR = sqrtf(totalR) / SCALE;
        totalG = sqrtf(totalG) / SCALE;
        totalB = sqrtf(totalB) / SCALE;

        const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
        t[channels * ((width_pad + 4) * y + x) + 2] = cR;
        t[channels * ((width_pad + 4) * y + x) + 1] = cG;
        t[channels * ((width_pad + 4) * y + x) + 0] = cB;
    }
}

int main(int argc, char** argv) {
    TIMING_START(total);
    assert(argc == 3);
    int deviceId;
    cudaGetDevice(&deviceId);
    DEBUG_PRINT("deviceId: %d\n", deviceId);

    unsigned height, width, channels, height_pad, width_pad;
    unsigned char* host_s = NULL;
    unsigned char* host_t = NULL;
    TIMING_START(read_png);
    read_png(argv[1], &host_s, &height, &width, &channels, &height_pad, &width_pad);
    TIMING_END(read_png);
    size_t totsize = channels * (width_pad + MASK_X - 1) * (height_pad + MASK_Y - 1) * sizeof(unsigned char);

    // dim3 blk(BLOCK_X, BLOCK_Y);
    // dim3 blk(1, BLOCK_Y);
    dim3 blk(BLOCK_X, 1);
    dim3 grid(width_pad / BLOCK_X, height_pad / BLOCK_Y);
    unsigned char* dev_s = NULL;
    unsigned char* dev_t = NULL;

    /* Profile */
    DEBUG_PRINT("%u %u %u %u\n", height, width, height_pad, width_pad);
    DEBUG_PRINT("%u %u %lu\n", height_pad + 4, width_pad + 4, totsize);

    /* Allocate device memory */
    CUDAEXE(cudaHostRegister(&host_s, totsize, cudaHostRegisterDefault));
    CUDAEXE(cudaMalloc(&dev_s, totsize));
    CUDAEXE(cudaMalloc(&dev_t, totsize));

    /* Copy host to device */
    CUDAEXE(cudaMemcpyAsync(dev_s, host_s, totsize, cudaMemcpyHostToDevice));
    CUDAEXE(cudaMemPrefetchAsync(dev_s, totsize, deviceId));

    /* Sobel */
    sobel<<<grid, blk>>>(dev_s, dev_t, height, width, channels, height_pad, width_pad);

    /* Copy device to host */
    host_t = (unsigned char*)malloc(totsize);
    CUDAEXE(cudaMemcpyAsync(host_t, dev_t, totsize, cudaMemcpyDeviceToHost));
    CUDAEXE(cudaGetLastError());
    CUDAEXE(cudaDeviceSynchronize());

    /* Output */
    TIMING_START(write_png);
    write_png(argv[2], host_t, height, width, channels, height_pad, width_pad);
    TIMING_END(write_png);

    /* Finalize */
    free(host_s);
    free(host_t);
    cudaFree(dev_s);
    cudaFree(dev_t);
    TIMING_END(total);
    return 0;
}
