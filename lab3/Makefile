NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v -arch=sm_61
# NVCCFLAGS += -ccbin g++ -Xcompiler -march=native
NVCCFLAGS += -ccbin clang++ -Xcompiler -Ofast -Xcompiler -march=native
NVCCFLAGS += -Xcompiler -Wall -Xcompiler -Wextra
# NVCCFLAGS += -DDEBUG
# NVCCFLAGS += -Xcompiler -g -Xcompiler -fsanitize=address
NVCC     = nvcc 
LDFLAGS = -lpng -lz

TARGETS = lab3

.PHONY: all
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
