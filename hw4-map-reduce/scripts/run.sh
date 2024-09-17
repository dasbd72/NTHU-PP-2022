#!bin/sh
make clean
make

JOB_NAME=TEST01

rm -f temp/*
mkdir outputs/$JOB_NAME
rm -f outputs/$JOB_NAME/*

srun -N 2 -c 5 ./mapreduce $JOB_NAME 12 4 testcases/01.word 2 testcases/01.loc ./outputs/$JOB_NAME