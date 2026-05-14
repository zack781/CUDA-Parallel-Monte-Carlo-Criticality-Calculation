```
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
```

## Hybrid CPU/GPU tail handling

The transport executable accepts optional run-size arguments:

```bash
./transport_sim [neutrons] [generations]
```

The GPU advances particles with event queues while the queue is large. When the active movement queue falls below a fixed CPU switch threshold, the remaining histories are copied to the host and completed with the scalar CPU history algorithm. CPU-produced fission sites are appended back to the fission bank before computing `k_eff` and resampling the next generation.
