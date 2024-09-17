#pragma once
#include <fstream>
#include <string>

#include "Logger.h"
#include "main.h"

class JobTracker {
   public:
    JobTracker(int argc, char** argv);
    ~JobTracker();

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

    Logger* logger;

    Tasks tasks;
    int kv_cnt;

    void ReadLocality();

    MapTaskData GetMapTask(int nodeID);
    void MapTaskDispatcher();

    void Shuffle();

    void ReduceTaskDispatcher();
};