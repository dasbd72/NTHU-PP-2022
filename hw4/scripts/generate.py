import os
import random
import numpy as np
from scipy.stats import rv_discrete

TESTCASES_DIR = "./testcases/"
JOB_NAME = "TEST00"
INPUT_FILE_NAME = TESTCASES_DIR + "00.word"
LOCALITY_CONFIG_FILENAME = TESTCASES_DIR + "00.loc"
TEMP_DIR = f"./temp"
OUTPUT_DIR = f"./outputs/{JOB_NAME}"

NODES = 4
CPUS = 2
NUM_REDUCER = 10
DELAY = 1
FILE_LENGTH = 40000
CHUNK_SIZE = 50

x = np.arange(1, NODES)
probs = [0.33, 0.33, 0.34]
# probs = [1, 0.0]
# probs = [0.9, 0.1]
# probs = [0.8, 0.2]
# probs = [0.7, 0.3]
# probs = [0.6, 0.4]
# probs = [0.5, 0.5]
# probs = [1]

with open(LOCALITY_CONFIG_FILENAME, "w") as file:
    for i in range(1, (FILE_LENGTH // CHUNK_SIZE) + 1):
        # file.write(f"{i} {random.randint(1, NODES - 1)}\n")
        file.write(f"{i} {np.random.choice(x, p=probs)}\n")

cmd = f'srun -N {NODES} -c {CPUS} ./mapreduce {JOB_NAME} {NUM_REDUCER} {DELAY} {INPUT_FILE_NAME} {CHUNK_SIZE} {LOCALITY_CONFIG_FILENAME} {OUTPUT_DIR}'
os.system('make clean')
os.system('make -j64')
os.system(f'rm -rf {TEMP_DIR}/*')
os.system(f'rm -rf {OUTPUT_DIR}')
os.system(f'mkdir {OUTPUT_DIR}')
os.system(cmd)