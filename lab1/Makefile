CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -march=native --fast-math
CFLAGS = -O3
TARGETS = lab1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) *.out

lab1: lab1.cc
	$(CXX) $(CXXFLAGS) -Wall -Wextra lab1.cc -o lab1