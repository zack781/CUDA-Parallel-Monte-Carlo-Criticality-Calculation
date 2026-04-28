#include "include/sim.cuh"
#include <curand_kernel.h>
#include <math.h>

#define PI 3.14159265358979323846f

// Initial neutron energies
// Initial angular position
// Initial radial position
// Direction angle at each new collision
// Free-flight distance
// Collision type selector
// Number of fission neutrons produced (2 or 3)

__global__
void init_rng(curandState *states, unsigned long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, tid, 0, &states[tid]);
}

__global__
void initialize_neutrons(
    curandState *states,
    Neutron *neutrons,
    float r_fuel,
    int N
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    curandState local_state = states[tid];

    // Initial position: uniformly distributed in fuel area
    float theta_pos = 2.0f * PI * curand_uniform(&local_state);
    float u = curand_uniform(&local_state);
    float r = r_fuel * sqrtf(u);

    neutrons[tid].x = r * cosf(theta_pos);
    neutrons[tid].y = r * sinf(theta_pos);

    // Initial direction: isotropic in 2D
    float theta_dir = 2.0f * PI * curand_uniform(&local_state);

    neutrons[tid].ux = cosf(theta_dir);
    neutrons[tid].uy = sinf(theta_dir);

    // Maxwell-like energy using 3 normal samples
    float a = curand_normal(&local_state);
    float b = curand_normal(&local_state);
    float c = curand_normal(&local_state);

    neutrons[tid].Energy = sqrtf(a * a + b * b + c * c);

    // Initial source is inside fuel
    neutrons[tid].region = 0;

    states[tid] = local_state;
}

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
