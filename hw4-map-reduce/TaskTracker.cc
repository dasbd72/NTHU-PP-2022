
#include "TaskTracker.h"

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

#include "main.h"

TaskTracker::TaskTracker(int argc, char** argv) {
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

    partitions.resize(num_reducer);
}
TaskTracker::~TaskTracker() {
}
void TaskTracker::Start() {
    MapTask();
    ReduceTask();
}

Records TaskTracker::InputSplit(int taskID) {
    Records records;
    std::ifstream input_file(input_filename);
    int start = taskID * chunk_size;
    std::string line;
    for (int i = 0; i < start; i++) {
        std::getline(input_file, line);
    }
    for (int i = 0; i < chunk_size; i++) {
        std::getline(input_file, line);
        records[i + start] = line;
    }
    input_file.close();
    return records;
}
WordCount TaskTracker::Map(Records& records) {
    WordCount wordCount;
    for (auto kv : records) {
        std::istringstream iss(kv.second);
        std::string token;
        while (std::getline(iss, token, ' ')) {
            wordCount[token]++;
        }
    }
    return wordCount;
}
void TaskTracker::Partition(WordCount& wordCount) {
    for (auto kv : wordCount) {
        int reducerID = hasher(kv.first) % num_reducer;
        pthread_mutex_lock(&partLock);
        partitions[reducerID][kv.first] += kv.second;
        pthread_mutex_unlock(&partLock);
    }
}
MapTaskData TaskTracker::GetMapTask(MapTaskRet prev) {
    MPI_Status status;
    MapTaskData task;
    pthread_mutex_lock(&taskLock);
    MPI_Send(&prev, sizeof(prev), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&task, sizeof(task), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    pthread_mutex_unlock(&taskLock);
    return task;
}
void* TaskTracker::MapThread(void* arg) {
    MapThread_t* data = (MapThread_t*)arg;
    TaskTracker* taskTracker = data->taskTracker;
    MapTaskRet prev = {-1, data->mapperID};
    while (true) {
        MapTaskData task;
        Records records;
        WordCount wordCount;

        task = taskTracker->GetMapTask(prev);
        if (task.nodeID == -1)
            break;
        if (task.nodeID != taskTracker->world_rank)
            sleep(taskTracker->delay);
        records = taskTracker->InputSplit(task.taskID);
        wordCount = taskTracker->Map(records);
        taskTracker->Partition(wordCount);

        prev.taskID = task.taskID;
    }
    return NULL;
}
void TaskTracker::MapTask() {
    DEBUG_MSG("MapTask")
    MapThread_t threads[num_cpu];
    WordCount totWordCount;
    int kv_cnt = 0, tmp;

    pthread_mutex_init(&taskLock, NULL);
    pthread_mutex_init(&partLock, NULL);
    for (int i = 0; i < num_cpu - 1; i++) {
        threads[i].taskTracker = this;
        threads[i].mapperID = (world_rank - 1) * (num_cpu - 1) + i;
        pthread_create(&threads[i].t, NULL, MapThread, (void*)&threads[i]);
    }
    for (int i = 0; i < num_cpu - 1; i++) {
        pthread_join(threads[i].t, NULL);
    }
    pthread_mutex_destroy(&taskLock);
    pthread_mutex_destroy(&partLock);

    for (int i = 0; i < num_reducer; i++) {
        std::string filename = std::to_string(i) + "-" + std::to_string(world_rank) + ".tmp";
        std::ofstream file(filename, std::ios_base::app);
        kv_cnt += partitions[i].size();
        for (auto kv : partitions[i]) {
            file << kv.first << " " << kv.second << "\n";
        }
        file.close();
    }
    MPI_Reduce(&kv_cnt, &tmp, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

Pairs TaskTracker::ReadIntermediate(int task) {
    Pairs pairs;
    std::string filename = std::to_string(task) + ".tmp";
    std::ifstream file(filename);
    std::string key;
    int val;
    while (file >> key >> val) {
        pairs.emplace_back(key, val);
    }
    file.close();
#ifdef REMOVE_INTERMEDIATE
    std::remove(filename.c_str());
#endif
    return pairs;
}
void TaskTracker::Sort(Pairs& pairs) {
    std::sort(pairs.begin(), pairs.end());
}
Groups TaskTracker::Group(Pairs& pairs) {
    Groups groups;
    for (auto kv : pairs) {
        groups[kv.first].push_back(kv.second);
    }
    return groups;
}
Results TaskTracker::Reduce(Groups& groups) {
    Results results;
    for (auto kv : groups) {
        for (auto v : kv.second) {
            results[kv.first] += v;
        }
    }
    return results;
}
void TaskTracker::Output(int task, Results& results) {
    std::string filename = output_dir + "/" + job_name + "-" + std::to_string(task) + ".out";
    std::ofstream file(filename);
    for (auto kv : results) {
        file << kv.first << " " << kv.second << "\n";
    }
    file.close();
}
int TaskTracker::GetReduceTask(int prev) {
    MPI_Status status;
    int task;
    MPI_Send(&prev, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&task, sizeof(task), MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    return task;
}
void TaskTracker::ReduceTask() {
    DEBUG_MSG("ReduceTask")
    int prev = -1;
    while (true) {
        Pairs pairs;
        Groups groups;
        Results results;

        int task = GetReduceTask(prev);
        if (task >= num_reducer)
            break;

        pairs = ReadIntermediate(task);
        Sort(pairs);
        groups = Group(pairs);
        results = Reduce(groups);
        Output(task, results);

        prev = task;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}