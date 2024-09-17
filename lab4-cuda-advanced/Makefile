NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v -arch=sm_61
NVCCFLAGS += -ccbin clang++ -Xcompiler "-Ofast -march=native -Wall -Wextra"
# NVCCFLAGS += -DDEBUG
# NVCCFLAGS += -DTIMING
# NVCCFLAGS += -Xcompiler "-g -fsanitize=address"
NVCC     = nvcc 
LDFLAGS = -lpng -lz

TARGETS = lab4

.PHONY: all
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
