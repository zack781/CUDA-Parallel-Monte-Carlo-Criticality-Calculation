CUDA-Parallel-Monte-Carlo-Criticality-Calculation/
├── include/
│   ├── sim.cuh          # struct definitions, constants, function declarations
├── src/
│   ├── main.cu          # power iteration loop, host code
│   ├── transport.cu     # neutron transport kernel
│   ├── fission_bank.cu  # normalize, resample, entropy kernels
│   ├── rng.cu           # cuRAND initialization kernel
├── tests/
│   └── test_sim.cu
├── Makefile
└── results/
    └── keff_history.csv
