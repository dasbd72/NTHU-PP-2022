#pragma once

#include <fstream>
#include <vector>

class Logger {
   public:
    Logger(int argc, char** argv);
    ~Logger();
    void SetMapTask(int num);
    void Start_Job();
    void Finish_Job();
    void Dispatch_MapTask(int taskID, int mapperID);
    void Complete_MapTask(int taskID);
    void Start_Shuffle(int n);
    void Finish_Shuffle();
    void Dispatch_ReduceTask(int taskID, int ID);
    void Complete_ReduceTask(int taskID);

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

    int num_map_task;
    int num_reduce_task;
    std::ofstream log_file;

    double start_job_time;
    std::vector<double> start_map_time;
    std::vector<double> start_reduce_time;

    long long start_shuffle_time;

    long long GetTime();
};