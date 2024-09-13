// #define DEBUG
#include <mpi.h>

#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>
#include <compare>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#define MAXN 536870911
#define MIN_PROC_N 1
#ifdef DEBUG
double tot_start_time, tot_duration, tot_sum_duration, tot_max_duration, tot_min_duration;
double nIO_start_time, nIO_duration, nIO_sum_duration, nIO_max_duration, nIO_min_duration;
double start_time, duration, sum_duration, max_duration, min_duration;
double sendrecv_start_time, sendrecv_duration, sum_sendrecv_duration, max_sendrecv_duration, min_sendrecv_duration;
double MPI_merge_start_time, MPI_merge_duration, sum_MPI_merge_duration, max_MPI_merge_duration, min_MPI_merge_duration;
#define MPI_EXECUTE(func)                                          \
    {                                                              \
        int rc = func;                                             \
        if (rc != MPI_SUCCESS) {                                   \
            printf("Error on MPI function at line %d.", __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, rc);                         \
        }                                                          \
    }
#define DEBUG_PRINT(fmt, args...)     \
    do {                              \
        fprintf(stderr, fmt, ##args); \
    } while (false);
#define TIMING_START() \
    start_time = MPI_Wtime();
#define TIMING_END(arg)                                                                                 \
    duration = MPI_Wtime() - start_time;                                                                \
    MPI_Reduce(&duration, &sum_duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);                    \
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);                    \
    MPI_Reduce(&duration, &min_duration, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);                    \
    if (world_rank == 0) {                                                                              \
        DEBUG_PRINT("%s, %lf, %lf, %lf\n", arg, sum_duration / world_size, max_duration, min_duration); \
    }
#define TOT_TIMING_START() \
    tot_start_time = MPI_Wtime();
#define TOT_TIMING_END()                                                                                          \
    tot_duration = MPI_Wtime() - tot_start_time;                                                                  \
    MPI_Reduce(&tot_duration, &tot_sum_duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);                      \
    MPI_Reduce(&tot_duration, &tot_max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);                      \
    MPI_Reduce(&tot_duration, &tot_min_duration, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);                      \
    if (world_rank == 0) {                                                                                        \
        DEBUG_PRINT("Total, %lf, %lf, %lf\n", tot_sum_duration / world_size, tot_max_duration, tot_min_duration); \
    }
#define NIO_TIMING_START() \
    nIO_start_time = MPI_Wtime();
#define NIO_TIMING_END()                                                                                                     \
    nIO_duration = MPI_Wtime() - nIO_start_time;                                                                             \
    MPI_Reduce(&nIO_duration, &nIO_sum_duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);                                 \
    MPI_Reduce(&nIO_duration, &nIO_max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);                                 \
    MPI_Reduce(&nIO_duration, &nIO_min_duration, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);                                 \
    if (world_rank == 0) {                                                                                                   \
        DEBUG_PRINT("Total Without IO, %lf, %lf, %lf\n", nIO_sum_duration / world_size, nIO_max_duration, nIO_min_duration); \
    }
#define SENDRECV_TIMING_START() sendrecv_duration = 0.0;
#define MERGE_TIMING_START() MPI_merge_duration = 0.0;
#define SENDRECV_FUNC(func)                                     \
    do {                                                        \
        sendrecv_start_time = MPI_Wtime();                      \
        func;                                                   \
        sendrecv_duration += MPI_Wtime() - sendrecv_start_time; \
    } while (false)
#define SENDRECV_TIMING_END()                                                                                                       \
    MPI_Reduce(&sendrecv_duration, &sum_sendrecv_duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);                              \
    MPI_Reduce(&sendrecv_duration, &max_sendrecv_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);                              \
    MPI_Reduce(&sendrecv_duration, &min_sendrecv_duration, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);                              \
    if (world_rank == 0) {                                                                                                          \
        DEBUG_PRINT("Sendrecv, %lf, %lf, %lf\n", sum_sendrecv_duration / world_size, max_sendrecv_duration, min_sendrecv_duration); \
    }
#define MERGE_FUNC(func)                                          \
    do {                                                          \
        MPI_merge_start_time = MPI_Wtime();                       \
        func;                                                     \
        MPI_merge_duration += MPI_Wtime() - MPI_merge_start_time; \
    } while (false)
#define MERGE_TIMING_END()                                                                                                            \
    MPI_Reduce(&MPI_merge_duration, &sum_MPI_merge_duration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);                              \
    MPI_Reduce(&MPI_merge_duration, &max_MPI_merge_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);                              \
    MPI_Reduce(&MPI_merge_duration, &min_MPI_merge_duration, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);                              \
    if (world_rank == 0) {                                                                                                            \
        DEBUG_PRINT("Merging, %lf, %lf, %lf\n", sum_MPI_merge_duration / world_size, max_MPI_merge_duration, min_MPI_merge_duration); \
    }
#else
#define MPI_EXECUTE(func) func
#define DEBUG_PRINT(fmt, args...)
#define SENDRECV_FUNC(func) func
#define MERGE_FUNC(func) func

#define TIMING_START()
#define TIMING_END(arg)
#define TOT_TIMING_START()
#define TOT_TIMING_END()
#define NIO_TIMING_START()
#define NIO_TIMING_END()
#define SENDRECV_TIMING_START()
#define MERGE_TIMING_START()
#define SENDRECV_FUNC(func) func
#define SENDRECV_TIMING_END()
#define MERGE_FUNC(func) func
#define MERGE_TIMING_END()
#endif

void MPI_merge_low(int ln, float *&larr, int rn, float *rarr, float *&tmparr);
void MPI_merge_high(int ln, float *larr, int rn, float *&rarr, float *&tmparr);
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int array_size = std::stoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    MPI_File input_file, output_file;
    int target_world_size;
    int step, remain, count, start;
    int odd_rank = -1, even_rank = -1;
    int odd_count = 0, even_count = 0;
    float recv_val;
    float *local_data = NULL, *recv_data = NULL, *temp_data = NULL;

    /* Calculate index */
    target_world_size = std::max(std::min(world_size, array_size / MIN_PROC_N), 1);
    step = array_size / target_world_size;
    remain = array_size % target_world_size;
    if (world_rank < remain) {
        count = step + 1;
        start = world_rank * (step + 1);
    } else if (world_rank < target_world_size) {
        count = step;
        start = remain * (step + 1) + (world_rank - remain) * step;
    } else {
        count = 0;
        start = array_size;
    }

    if (world_rank & 1) {
        odd_rank = world_rank + 1;
        even_rank = world_rank - 1;
    } else {
        odd_rank = world_rank - 1;
        even_rank = world_rank + 1;
    }
    if (odd_rank < 0 || odd_rank >= target_world_size || world_rank >= target_world_size) {
        odd_rank = MPI_PROC_NULL;
    }
    if (even_rank < 0 || even_rank >= target_world_size || world_rank >= target_world_size) {
        even_rank = MPI_PROC_NULL;
    }

    if (odd_rank == MPI_PROC_NULL) {
        odd_count = 0;
    } else {
        if (odd_rank < remain)
            odd_count = step + 1;
        else
            odd_count = step;
    }

    if (even_rank == MPI_PROC_NULL) {
        even_count = 0;
    } else {
        if (even_rank < remain)
            even_count = step + 1;
        else
            even_count = step;
    }

    TOT_TIMING_START();
    // DEBUG_PRINT("rank: %d %d %d, count: %d %d %d\n", world_rank, odd_rank, even_rank, count, odd_count, even_count);
    /* Read input */
    TIMING_START();
    local_data = new float[step + 1];
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if (count != 0) {
        MPI_File_read_at(input_file, sizeof(float) * start, local_data, count, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&input_file);
    TIMING_END("Read IO");
    NIO_TIMING_START();

    /* Local sort */
    TIMING_START();
    if (count != 0) {
        // std::sort(local_data, local_data + count);
        // boost::sort::pdqsort(local_data, local_data + count);
        // boost::sort::spreadsort::float_sort(local_data, local_data + count);
        boost::sort::spreadsort::spreadsort(local_data, local_data + count);
    }
    TIMING_END("Local Sort");

    /* Sorting */
    SENDRECV_TIMING_START();
    MERGE_TIMING_START();
    TIMING_START();
    temp_data = new float[step + 1];
    recv_data = new float[step + 1];
    for (int p = 0; p < target_world_size + 1; p++) {
        if (p & 1) {
            if (odd_rank != MPI_PROC_NULL) { /* Odd phase */
                if (world_rank & 1) {
                    SENDRECV_FUNC(MPI_Sendrecv(local_data + count - 1, 1, MPI_FLOAT, odd_rank, 0, &recv_val, 1, MPI_FLOAT, odd_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    if (local_data[count - 1] > recv_val) {
                        SENDRECV_FUNC(MPI_Sendrecv(local_data, count, MPI_FLOAT, odd_rank, 0, recv_data, odd_count, MPI_FLOAT, odd_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                        MERGE_FUNC(MPI_merge_low(count, local_data, odd_count, recv_data, temp_data));
                    }
                } else {
                    SENDRECV_FUNC(MPI_Sendrecv(local_data, 1, MPI_FLOAT, odd_rank, 0, &recv_val, 1, MPI_FLOAT, odd_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    if (local_data[0] < recv_val) {
                        SENDRECV_FUNC(MPI_Sendrecv(local_data, count, MPI_FLOAT, odd_rank, 0, recv_data, odd_count, MPI_FLOAT, odd_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                        MERGE_FUNC(MPI_merge_high(odd_count, recv_data, count, local_data, temp_data));
                    }
                }
            }
        } else {
            if (even_rank != MPI_PROC_NULL) { /* Even phase */
                if (world_rank & 1) {
                    SENDRECV_FUNC(MPI_Sendrecv(local_data, 1, MPI_FLOAT, even_rank, 0, &recv_val, 1, MPI_FLOAT, even_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    if (local_data[0] < recv_val) {
                        SENDRECV_FUNC(MPI_Sendrecv(local_data, count, MPI_FLOAT, even_rank, 0, recv_data, even_count, MPI_FLOAT, even_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                        MERGE_FUNC(MPI_merge_high(even_count, recv_data, count, local_data, temp_data));
                    }
                } else {
                    SENDRECV_FUNC(MPI_Sendrecv(local_data + count - 1, 1, MPI_FLOAT, even_rank, 0, &recv_val, 1, MPI_FLOAT, even_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    if (local_data[count - 1] > recv_val) {
                        SENDRECV_FUNC(MPI_Sendrecv(local_data, count, MPI_FLOAT, even_rank, 0, recv_data, even_count, MPI_FLOAT, even_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                        MERGE_FUNC(MPI_merge_low(count, local_data, even_count, recv_data, temp_data));
                    }
                }
            }
        }
    }
    TIMING_END("Global sort");
    SENDRECV_TIMING_END();
    MERGE_TIMING_END();
    NIO_TIMING_END();
    /* Write output */
    TIMING_START();
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (count != 0) {
        MPI_File_write_at(output_file, sizeof(float) * start, local_data, count, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file);
    TIMING_END("Output IO");
    TOT_TIMING_END();

    /* Finalize program */
    delete[] temp_data;
    delete[] recv_data;
    delete[] local_data;
    MPI_Finalize();
    return 0;
}

void MPI_merge_low(int ln, float *&larr, int rn, float *rarr, float *&tmparr) {
    int li, ri, ti;
    li = ri = ti = 0;
    while (li < ln && ri < rn && ti < ln) {
        if (larr[li] <= rarr[ri]) {
            tmparr[ti++] = larr[li++];
        } else {
            tmparr[ti++] = rarr[ri++];
        }
    }
    while (li < ln && ti < ln) {
        tmparr[ti++] = larr[li++];
    }
    while (ri < rn && ti < ln) {
        tmparr[ti++] = rarr[ri++];
    }
    std::swap(larr, tmparr);
    return;
}

void MPI_merge_high(int ln, float *larr, int rn, float *&rarr, float *&tmparr) {
    int li, ri, ti;
    li = ln - 1;
    ti = ri = rn - 1;
    while (li >= 0 && ri >= 0 && ti >= 0) {
        if (larr[li] <= rarr[ri]) {
            tmparr[ti--] = rarr[ri--];
        } else {
            tmparr[ti--] = larr[li--];
        }
    }
    while (li >= 0 && ti >= 0) {
        tmparr[ti--] = larr[li--];
    }
    while (ri >= 0 && ti >= 0) {
        tmparr[ti--] = rarr[ri--];
    }
    std::swap(rarr, tmparr);
    return;
}