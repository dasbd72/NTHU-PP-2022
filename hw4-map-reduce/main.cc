#include "main.h"

#include <mpi.h>
#include <pthread.h>

#include <cassert>

#include "JobTracker.h"
#include "TaskTracker.h"

int main(int argc, char** argv) {
    assert(argc == 8);

    int world_rank;
    int world_size;
    // MPI init
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        // Jobtracker
        JobTracker jobTracker(argc, argv);
        jobTracker.Start();
    } else {
        // Tasktracker
        TaskTracker taskTracker(argc, argv);
        taskTracker.Start();
    }

    // MPI finalize
    MPI_Finalize();

    return 0;
}