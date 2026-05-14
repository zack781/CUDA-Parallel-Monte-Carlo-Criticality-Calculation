# CUDA Parallel Monte Carlo Criticality Calculation

A GPU-accelerated Monte Carlo neutron transport simulator for 2D nuclear reactor criticality analysis. The code estimates the effective neutron multiplication factor (**k-eff**) of a fuel pin cell using a flattened event-queue kernel design that minimizes thread divergence on NVIDIA GPUs.

## Physics Background

The simulator models a 2D fuel pin cell — a cylindrical UO₂ fuel pellet surrounded by aluminum cladding inside a water moderator. Reflective boundary conditions on the moderator cell walls simulate an infinite lattice of identical pins.

Criticality is determined by the **k-effective** (k_eff): the ratio of neutrons produced in one generation to those produced in the previous one. A value of 1.0 means the chain reaction is self-sustaining.

**Geometry:**

| Region    | Material          | Outer radius |
|-----------|-------------------|-------------|
| Fuel      | 3% enriched UO₂   | 0.53 cm     |
| Cladding  | Aluminum          | 0.90 cm     |
| Moderator | H₂O               | 0.9185 cm (half-pitch) |

**Nuclear data:** Multi-group cross-sections for 10 energy groups spanning 3×10⁻⁸ to 3×10¹ MeV, generated with OpenMC's multigroup library. Three reaction types are modeled: fission, radiative capture, and elastic scattering.

## Algorithm

The simulator uses a **flattened event-based** design rather than the conventional one-thread-per-history approach. Each particle occupies a slot in one of two queues and waits for a single event to process, eliminating the long, uneven per-history execution that causes thread divergence.

```
move_queue = source_particles
collision_queue = empty
fission_bank = empty

while queues not empty:
    move_kernel(move_queue)         // transport to boundary or collision site
    collision_kernel(collision_queue) // sample reaction, route particle
    compact / sort queues

k_eff = |fission_bank| / N
source_particles = resample(fission_bank)  // next generation source
```

**move_kernel** — for each particle, samples a path length from Beer-Lambert attenuation, computes distance to the nearest boundary (ray-circle for fuel/clad, ray-slab for moderator walls), and routes the particle to `collision_queue` (collision first) or back to `move_queue` (boundary first after updating its region).

**collision_kernel** — samples a reaction from the local cross-section mix. Fission produces 2–3 secondary neutrons (added to `fission_bank` via warp-synchronous cooperative insertion). Scatter updates energy and direction. Capture kills the particle.

### Hybrid CPU/GPU Tail

As a generation winds down, the active queue drains below a threshold (`CPU_SWITCH_THRESHOLD = 4096`). At that point the remaining particles are copied to the host and completed with a scalar history-based CPU loop. The CPU-produced fission sites are merged back into the bank before computing k_eff and resampling.

This avoids launching near-empty GPU grids with poor occupancy for the tail of each generation.

### Key GPU Optimizations

- **Struct-of-Arrays (SoA) layout** for coalesced global memory access across threads
- **Constant memory** for the 10-group cross-section tables, reducing L1 pressure
- **Region sorting** of the move queue every 5 iterations to improve spatial locality
- **Lock-free atomic queue insertion** via `atomicCAS` with overflow detection
- **Warp-synchronous fission bank insertion** using `__ballot_sync` / `__shfl_sync` primitives

## Project Structure

```
├── include/
│   └── sim.cuh          # structs (NeutronSoA, XS, Tallies), constants, kernel declarations
├── src/
│   ├── main.cu          # power iteration loop, host memory management, hybrid tail
│   ├── transport.cu     # move_kernel and collision_kernel
│   ├── fission_bank.cu  # resample_kernel for next-generation source
│   ├── rng.cu           # cuRAND initialization, neutron birth sampling
│   ├── cpu_sim.cpp      # standalone CPU reference implementation
│   ├── history_main.cu  # experimental one-thread-per-history GPU kernel
│   └── common.h         # CUDA error-checking macros
├── gen_xs.py            # generate multi-group cross-sections via OpenMC
├── find_boron.py        # bisection search for critical boron concentration
├── materials.xml        # OpenMC material definitions (reference)
├── settings.xml         # OpenMC simulation settings (reference)
├── tallies.xml          # OpenMC tally definitions (reference)
├── transport_algo.md    # algorithm design notes
└── Makefile
```

## Requirements

- NVIDIA GPU, compute capability ≥ 8.0 (A100, L40S, H100)
- CUDA Toolkit 11.0+
- C++17-capable compiler (GCC, Clang, or NVIDIA HPC SDK)
- CUDA Thrust (bundled with the toolkit)
- OpenMC 0.13+ with `openmc.mgxs` (only needed to regenerate cross-sections)

The Makefile is configured for the Perlmutter HPC system (NERSC). Adjust `nvcc` flags and the `CC` wrapper for other environments.

## Building

```bash
# GPU event-queue transport (primary)
make transport_sim

# Experimental history-based GPU kernel
make history_sim

# CPU-only reference
make cpu_sim
```

## Usage

```bash
# GPU simulation — defaults: 10,000 neutrons, 10 generations
./transport_sim [neutrons] [generations]

# CPU reference
./cpu_sim [neutrons] [generations] [boron_ppm]

# Find critical boron concentration (bisection, uses cpu_sim)
python find_boron.py
```

**Example output:**
```
Generation 1: fission bank = 9873, k_eff = 1.04521
Generation 2: fission bank = 10241, k_eff = 1.02187
...
--- Final Tallies ---
Total interactions : 1482031
Scattering events  : 893204
Capture events     : 371844
Fission events     : 214983
Leakage            : 2000
Neutrons produced  : 494221
Average nu         : 2.298
```

The simulation also writes `keff_history.csv` with the per-generation k_eff for convergence analysis.

## Cross-Section Generation

Cross-sections are hard-coded in `include/sim.cuh` as constant-memory arrays. To regenerate them from first principles:

```bash
python gen_xs.py   # requires OpenMC with ENDF/B-VIII.0 data
```

## References

- Islam, ASM Fakhrul. *Monte Carlo Criticality Calculation for a PWR Pin Cell*. NCSU, 2019. (reference OpenMC model in `QualifyingMC/`)
- OpenMC Development Team. [openmc.org](https://openmc.org)
- NVIDIA. *CUDA C++ Programming Guide*, Chapter 9 (Warp-Level Primitives).
