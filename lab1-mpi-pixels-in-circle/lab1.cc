#include <assert.h>
#include <math.h>
#include <mpi.h>
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
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;
    unsigned long long local_pixels = 0;
    unsigned long long target = ceil(double(r) * 0.70710678118);
    unsigned long long stride = ceil(double(r - target) / double(world_size));
    unsigned long long start = world_rank * stride + target;
    unsigned long long end = min(start + stride, r);

    for (unsigned long long x = start; x < end; ++x) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        local_pixels += y;
        local_pixels %= k;
    }

    // MPI_Reduce(&local_pixels, &pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (world_rank == 0) {
        pixels = local_pixels;
        unsigned long long tmp;
        for(int i = 1; i < world_size; i++) {
            MPI_Recv(&tmp, 1, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pixels = (pixels + tmp) % k;
        }
    } else {
        MPI_Send(&local_pixels, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        pixels %= k;
        pixels = pixels * 2 + target * target;
        pixels %= k;
        printf("%llu\n", (4 * pixels) % k);
    }

    MPI_Finalize();
}
