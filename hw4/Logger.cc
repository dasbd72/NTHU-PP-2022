#include "Logger.h"

#include <mpi.h>

#include <chrono>
#include <iostream>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

Logger::Logger(int argc, char** argv) {
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

    num_reduce_task = num_reducer;
    start_reduce_time.resize(num_reduce_task);

    log_file.open(log_filename);
}

Logger::~Logger() {
    log_file.close();
}

void Logger::SetMapTask(int num) {
    num_map_task = num;
    start_map_time.resize(num_map_task);
}

long long Logger::GetTime() {
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void Logger::Start_Job() {
    start_job_time = MPI_Wtime();
    log_file << GetTime() << ",";
    log_file << "Start_Job,";
    log_file << job_name << ",";
    log_file << world_size << ",";
    log_file << num_cpu << ",";
    log_file << num_reducer << ",";
    log_file << delay << ",";
    log_file << input_filename << ",";
    log_file << chunk_size << ",";
    log_file << locality_config_filename << ",";
    log_file << output_dir << "\n";
}

void Logger::Finish_Job() {
    auto curr_time = MPI_Wtime();
    log_file << GetTime() << ",";
    log_file << "Finish_Job,";
    log_file << curr_time - start_job_time << "\n";
}

void Logger::Dispatch_MapTask(int taskID, int mapperID) {
    start_map_time[taskID] = MPI_Wtime();
    log_file << GetTime() << ",";
    log_file << "Dispatch_MapTask,";
    log_file << taskID + 1 << ",";
    log_file << mapperID + 1 << "\n";
}

void Logger::Complete_MapTask(int taskID) {
    auto curr_time = MPI_Wtime();
    log_file << GetTime() << ",";
    log_file << "Complete_MapTask,";
    log_file << taskID + 1 << ",";
    log_file << curr_time - start_map_time[taskID] << "\n";
}

void Logger::Start_Shuffle(int n) {
    start_shuffle_time = GetTime();
    log_file << start_shuffle_time << ",";
    log_file << "Start_Shuffle,";
    log_file << n << "\n";
}

void Logger::Finish_Shuffle() {
    auto curr_time = GetTime();
    log_file << curr_time << ",";
    log_file << "Finish_Shuffle,";
    log_file << curr_time - start_shuffle_time << "\n";
}

void Logger::Dispatch_ReduceTask(int taskID, int reducerID) {
    start_reduce_time[taskID] = MPI_Wtime();
    log_file << GetTime() << ",";
    log_file << "Dispatch_ReduceTask,";
    log_file << taskID + 1 << ",";
    log_file << reducerID << "\n";
}

void Logger::Complete_ReduceTask(int taskID) {
    auto curr_time = MPI_Wtime();
    log_file << GetTime() << ",";
    log_file << "Complete_ReduceTask,";
    log_file << taskID + 1 << ",";
    log_file << curr_time - start_reduce_time[taskID] << "\n";
}