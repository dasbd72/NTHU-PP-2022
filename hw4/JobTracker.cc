#include "JobTracker.h"

#include <mpi.h>

#include <fstream>
#include <iostream>

#include "main.h"

JobTracker::JobTracker(int argc, char** argv) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set);
    num_cpu = CPU_COUNT(&cpu_set);

    job_name = argv[1];
    num_reducer = std::stoi(argv[2]);
    delay = std::stoi(argv[3]);
    input_filename = argv[4];
    chunk_size = std::stoi(argv[5]);
    locality_config_filename = argv[6];
    output_dir = argv[7];
    log_filename = output_dir + "/" + job_name + "-" + "log.out";

    logger = new Logger(argc, argv);
    kv_cnt = 0;
}

JobTracker::~JobTracker() {
    delete logger;
}

void JobTracker::Start() {
    logger->Start_Job();
    ReadLocality();
    MapTaskDispatcher();
    Shuffle();
    ReduceTaskDispatcher();
    logger->Finish_Job();
}

void JobTracker::ReadLocality() {
    DEBUG_MSG("ReadLocality");
    std::ifstream locality_config_file(locality_config_filename);
    tasks.resize(world_size);
    int num_chunks = 0;
    for (int chunkID, nodeID, taskID; locality_config_file >> chunkID >> nodeID;) {
        if (nodeID - 1 >= world_size - 1)
            nodeID = (nodeID - 1) % (world_size - 1) + 1;
        taskID = chunkID - 1;
        tasks[nodeID].push_back(taskID);
        num_chunks = std::max(chunkID, num_chunks);
    }
    locality_config_file.close();
    logger->SetMapTask(num_chunks);
}

MapTaskData JobTracker::GetMapTask(int nodeID) {
    MapTaskData task;
    if (!tasks[nodeID].empty()) {
        task.taskID = tasks[nodeID].front();
        task.nodeID = nodeID;
        tasks[nodeID].pop_front();
    } else {
        int minChunkID = 0x7fffffff;
        int minNodeID = -1;
        for (int i = 1; i < world_size; i++) {
            if (!tasks[i].empty()) {
                if (tasks[i].front() < minChunkID) {
                    minChunkID = tasks[i].front();
                    minNodeID = i;
                }
            }
        }
        task.nodeID = minNodeID;
        if (minNodeID != -1) {
            task.taskID = tasks[minNodeID].front();
            tasks[minNodeID].pop_front();
        }
    }
    return task;
}

void JobTracker::MapTaskDispatcher() {
    DEBUG_MSG("MapTaskDispatcher");
    MPI_Status status;
    MapTaskData task;
    MapTaskRet retTask;
    int tmp = 0;
    int remainThreads = (num_cpu - 1) * (world_size - 1);

    while (remainThreads > 0) {
        MPI_Recv(&retTask, sizeof(retTask), MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if (retTask.taskID != -1)
            logger->Complete_MapTask(retTask.taskID);
        task = GetMapTask(status.MPI_SOURCE);
        MPI_Send(&task, sizeof(MapTaskData), MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        if (task.nodeID == -1)
            remainThreads--;
        else
            logger->Dispatch_MapTask(task.taskID, retTask.mapperID);
    }
    MPI_Reduce(&tmp, &kv_cnt, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

void JobTracker::Shuffle() {
    DEBUG_MSG("Shuffle");
    logger->Start_Shuffle(kv_cnt);
    for (int i = 0; i < num_reducer; i++) {
        Pairs pairs;
        for (int r = 1; r < world_size; r++) {
            std::string filename = std::to_string(i) + "-" + std::to_string(r) + ".tmp";
            std::ifstream file(filename);
            std::string key;
            int val;
            while (file >> key >> val) {
                kv_cnt++;
                pairs.emplace_back(key, val);
            }
            file.close();
#ifdef REMOVE_INTERMEDIATE
            std::remove(filename.c_str());
#endif
        }
        std::string filename = std::to_string(i) + ".tmp";
        std::ofstream file(filename);
        for (auto kv : pairs) {
            file << kv.first << " " << kv.second << "\n";
        }
        file.close();
    }
    logger->Finish_Shuffle();
}

void JobTracker::ReduceTaskDispatcher() {
    DEBUG_MSG("ReduceTaskDispatcher");
    MPI_Status status;
    int retTask;
    int task = 0, curr = 0;
    int remainThreads = world_size - 1;
    while (remainThreads > 0) {
        MPI_Recv(&retTask, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if (retTask != -1)
            logger->Complete_ReduceTask(retTask);
        task = curr++;
        MPI_Send(&task, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        if (task >= num_reducer)
            remainThreads--;
        else
            logger->Dispatch_ReduceTask(task, status.MPI_SOURCE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}