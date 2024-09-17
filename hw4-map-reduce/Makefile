CC = gcc
CXX = g++
CC := mpicc -cc=$(CC)
CXX := mpicxx -cxx=$(CXX)
CFLAGS = -lm -O3 -fopenmp -pthread -march=native
CFLAGS += -DDEBUG
CFLAGS += -DREMOVE_INTERMEDIATE
# CFLAGS += -fsanitize=address -g
CFLAGS += -Wall -Wextra
CXXFLAGS = -std=c++17 $(CFLAGS)

TARGETS = mapreduce
DEP = main.cc JobTracker.cc TaskTracker.cc Logger.cc
OBJ = main.o JobTracker.o TaskTracker.o Logger.o

.PHONY: all
all: $(TARGETS)

mapreduce: $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

main.o: main.cc
	$(CXX) $(CXXFLAGS) -c $^

JobTracker.o: JobTracker.cc
	$(CXX) $(CXXFLAGS) -c $^

TaskTracker.o: TaskTracker.cc
	$(CXX) $(CXXFLAGS) -c $^

Logger.o: Logger.cc
	$(CXX) $(CXXFLAGS) -c $^

.PHONY: clean
clean:
	rm -f $(TARGETS) $(OBJ)
