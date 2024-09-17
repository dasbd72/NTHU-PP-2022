import os
import sys
import json

argc = len(sys.argv)
assert(argc > 1)
with open(sys.argv[1]) as file:
    data = json.load(file)

TESTCASES_DIR = "./testcases/"
NODES = data["NODES"]
CPUS = data["CPUS"]
JOB_NAME = data["JOB_NAME"]
NUM_REDUCER = data["NUM_REDUCER"]
DELAY = data["DELAY"]
INPUT_FILE_NAME = TESTCASES_DIR + data["INPUT_FILE_NAME"]
CHUNK_SIZE = data["CHUNK_SIZE"]
LOCALITY_CONFIG_FILENAME = TESTCASES_DIR + data["LOCALITY_CONFIG_FILENAME"]
OUTPUT_DIR = f"./outputs/{JOB_NAME}"
TEMP_DIR = f"./temp"

cmd = f'srun -N {NODES} -c {CPUS} ./mapreduce {JOB_NAME} {NUM_REDUCER} {DELAY} {INPUT_FILE_NAME} {CHUNK_SIZE} {LOCALITY_CONFIG_FILENAME} {OUTPUT_DIR}'
os.system('make clean')
os.system('make -j64')
os.system(f'rm -rf {TEMP_DIR}/*')
os.system(f'rm -rf {OUTPUT_DIR}')
os.system(f'mkdir {OUTPUT_DIR}')
os.system(cmd)