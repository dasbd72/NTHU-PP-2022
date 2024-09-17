#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>

unsigned long long min(unsigned long long a, unsigned long long b) {
    if (a < b)
        return a;
    else
        return b;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }
    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long rr = r * r;
    unsigned long long pixels = 0;
    unsigned long long target = ceil(double(r) * 0.70710678118);
    unsigned long long ncpus = omp_get_max_threads();

#pragma omp parallel for num_threads(ncpus) schedule(static) default(shared) reduction(+ \
                                                                                       : pixels)
    for (unsigned long long x = target; x < r; x++) {
        unsigned long long y = ceil(sqrtl(rr - x * x));
        pixels += y;
        if(pixels >= k)
            pixels %= k;
    }

    //     unsigned long long stride = ceil(double(r - target) / double(ncpus));
    // #pragma omp parallel for num_threads(ncpus) schedule(static)
    //     for (unsigned long long cpuid = 0; cpuid < ncpus; cpuid++) {
    //         unsigned long long start = cpuid * stride + target;
    //         unsigned long long end = min(start + stride, r);
    //         unsigned long long local_pixels = 0;
    //         for (unsigned long long x = start; x < end; x++) {
    //             unsigned long long y = ceil(sqrtl(rr - x * x));
    //             local_pixels += y;
    //             local_pixels %= k;
    //         }
    // #pragma omp critical
    //         pixels = (pixels + local_pixels) % k;
    //     }

    pixels = pixels * 2 + target * target;
    pixels %= k;
    printf("%llu\n", (4 * pixels) % k);
}
