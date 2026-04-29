# Compiler and flags for Perlmutter
NVCC = nvcc
CXX = CC  # Cray compiler wrapper (automatically picks the right compiler)

# CUDA compute capability for A100
CUDA_ARCH = -gencode arch=compute_80,code=sm_80

# Compilation flags
CFLAGS = -std=c++17 -O3
NVCC_FLAGS = $(CUDA_ARCH) $(CFLAGS) -I. -I./include

# Source files
SOURCES = src/main.cu
HEADERS = include/sim.cuh src/common.h src/transport.cu src/rng.cu src/fission_bank.cu

# Output executable
TARGET = transport_sim

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SOURCES)

# Run on GPU
run: $(TARGET)
	srun -n 1 --gpus=1 ./$(TARGET) $(ARGS)

# Clean
clean:
	rm -f $(TARGET) *.o

# Verbose build (shows actual compilation commands)
verbose:
	$(NVCC) $(NVCC_FLAGS) -v -o $(TARGET) $(SOURCES)

.PHONY: all run clean verbose
