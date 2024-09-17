#pragma once
#include <string>

#include "main.h"

class TaskTracker {
   public:
    TaskTracker(int argc, char** argv);
    ~TaskTracker();

    void Start();

   private:
    int world_rank;
    int world_size;
    cpu_set_t cpu_set;
    int num_cpu;

    std::string job_name;
    int num_reducer;
    int delay;
    std::string input_filename;
    int chunk_size;
    std::string locality_config_filename;
    std::string output_dir;
    std::string log_filename;

    pthread_mutex_t taskLock;
    pthread_mutex_t partLock;
    std::hash<std::string> hasher;
    Partitions partitions;

    Records InputSplit(int taskID);
    WordCount Map(Records& records);
    void Partition(WordCount& wordCount);
    static void* MapThread(void* arg);
    void MapTask();

    Pairs ReadIntermediate(int task);
    void Sort(Pairs& pairs);
    Groups Group(Pairs& pairs);
    Results Reduce(Groups& groups);
    void Output(int task, Results& results);
    void ReduceTask();
    MapTaskData GetMapTask(MapTaskRet prev);
    int GetReduceTask(int prev);
};

class MapThread_t {
   public:
    MapThread_t() {}
    ~MapThread_t() {}

    pthread_t t;
    int mapperID;
    TaskTracker* taskTracker;
};