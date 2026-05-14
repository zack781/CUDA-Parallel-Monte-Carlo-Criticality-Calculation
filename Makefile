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
HISTORY_SOURCES = src/history_main.cu

# Output executable
TARGET     = transport_sim
HISTORY_TARGET = history_sim
CPU_TARGET = cpu_sim

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SOURCES)

$(HISTORY_TARGET): $(HISTORY_SOURCES) include/sim.cuh src/common.h src/rng.cu src/fission_bank.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $(HISTORY_SOURCES)

# Run on GPU
run: $(TARGET)
	srun -n 1 --gpus=1 ./$(TARGET) $(ARGS)

# CPU-only simulation (no CUDA required)
$(CPU_TARGET): src/cpu_sim.cpp
	$(CXX) -std=c++17 -O3 -o $@ $<

cpu: $(CPU_TARGET)

history: $(HISTORY_TARGET)

# Clean
clean:
	rm -f $(TARGET) $(HISTORY_TARGET) $(CPU_TARGET) *.o

# Verbose build (shows actual compilation commands)
verbose:
	$(NVCC) $(NVCC_FLAGS) -v -o $(TARGET) $(SOURCES)

.PHONY: all run cpu history clean verbose
