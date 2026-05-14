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

## Tail cutoff correction

The transport executable accepts optional tail-truncation arguments:

```bash
./transport_sim [neutrons] [generations] [batch_size] [tail_fraction]
```

`tail_fraction` stops a generation once the active movement queue falls below that fraction of the original source population. For example, `0.001` truncates when fewer than `0.1%` of the source particles remain active.

Tail yield is estimated dynamically from the current generation. Completed particles accumulate observed fission-bank production by terminal region, remaining tail particles are counted by their current region, and the corrected `k_eff` adds the region-weighted expected tail fissions. If a region has no completed particles in the current generation, it falls back to the generation-wide observed yield.

The correction adjusts the scalar reported `k_eff`, but the next-generation source is still resampled only from the explicit fission bank.
