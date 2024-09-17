#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>

unsigned long long min(unsigned long long a, unsigned long long b) {
    if (a < b)
        return a;
    else
        return b;
}

typedef struct GlobalData {
    unsigned long long r;
    unsigned long long k;
    unsigned long long target;
    unsigned long long stride;
} GlobalData;

typedef struct Data {
    unsigned long long cpuid;
    unsigned long long pixels;
    GlobalData* globalData;
} Data;

void func(void* arg) {
    Data* data = (Data*)arg;
    unsigned long long rr = data->globalData->r * data->globalData->r;
    unsigned long long start = data->cpuid * data->globalData->stride + data->globalData->target;
    unsigned long long end = min(start + data->globalData->stride, data->globalData->r);
    data->pixels = 0;

    for (unsigned long long x = start; x < end; x++) {
        unsigned long long y = ceil(sqrtl(rr - x * x));
        data->pixels += y;
        if (data->pixels >= data->globalData->k)
            data->pixels %= data->globalData->k;
    }
}

void* thread_func(void* arg) {
    func(arg);
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        return 1;
    }
    GlobalData globalData;
    globalData.r = atoll(argv[1]);
    globalData.k = atoll(argv[2]);
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    unsigned long long ncpus = CPU_COUNT(&cpuset);
    unsigned long long pixels = 0;
    globalData.target = ceil(double(globalData.r) * 0.70710678118);
    globalData.stride = ceil(double(globalData.r - globalData.target) / double(ncpus));
    Data data_arr[ncpus];
    pthread_t threads[ncpus];

    for (unsigned long long cpuid = 1; cpuid < ncpus; cpuid++) {
        data_arr[cpuid].cpuid = cpuid;
        data_arr[cpuid].globalData = &globalData;
        pthread_create(&threads[cpuid], NULL, thread_func, &data_arr[cpuid]);
    }

    data_arr[0].cpuid = 0;
    data_arr[0].globalData = &globalData;
    func(&data_arr[0]);

    for (unsigned long long cpuid = 1; cpuid < ncpus; cpuid++) {
        pthread_join(threads[cpuid], NULL);
    }

    for (unsigned long long cpuid = 0; cpuid < ncpus; cpuid++) {
        pixels += data_arr[cpuid].pixels;
        pixels %= globalData.k;
    }

    pixels = pixels * 2 + globalData.target * globalData.target;
    pixels %= globalData.k;
    printf("%llu\n", (4 * pixels) % globalData.k);
}
