#include "../include/sim.cuh"

// Initial neutron energies
// Initial angular position
// Initial radial position
// Direction angle at each new collision
// Free-flight distance
// Collision type selector
// Number of fission neutrons produced (2 or 3)

__device__ float random_uniform(unsigned int *state)
{
    // TODO: Replace with cuRAND or a higher-quality RNG if needed.
}

__device__ void sample_isotropic_direction(unsigned int *state, float *ux, float *uy)
{
    // TODO: Sample an isotropic 2D direction using random_uniform().
}

__global__ void init_rng_kernel()
{
    // TODO: Initialize one RNG state per CUDA thread/history.
}

void rng_wrapper()
{
    // TODO: Allocate, initialize, and manage RNG state buffers from host code.
}
