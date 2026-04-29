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
void init_rng(curandState *states, unsigned long seed, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= 0 && tid >= n) return;

    curand_init(seed, tid, 0, &states[tid]);
}

__global__
void initialize_neutrons(
    curandState *states,
    Neutron *neutrons,
    float r_fuel,
    int n
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= n) return;

    curandState local_state = states[tid];

    // Initial position: uniformly distributed in fuel area
    float theta_pos = 2.0f * PI * random_uniform(&local_state);
    float u = random_uniform(&local_state);
    float r = r_fuel * sqrtf(u);

    neutrons[tid].x = r * cosf(theta_pos);
    neutrons[tid].y = r * sinf(theta_pos);

    // Direction is sampled in move_kernel when regionchange == 0.
    neutrons[tid].ux = 1.0f;
    neutrons[tid].uy = 0.0f;
    neutrons[tid].Energy = sample_initial_energy(&local_state);

    // Initial source is inside fuel
    neutrons[tid].region = FUEL;
    neutrons[tid].regionchange = 0;
    neutrons[tid].rng_state = local_state;

    states[tid] = local_state;
}

__device__ float random_uniform(curandState *state)
{
    return curand_uniform(state);
}

__device__ void sample_isotropic_direction(curandState *state, float *ux, float *uy)
{
    float theta = 2.0f * PI * random_uniform(state);
    *ux = cosf(theta);
    *uy = sinf(theta);
}

__device__ float sample_initial_energy(curandState *state)
{
    // Maxwell-like speed distribution from three standard normal components.
    float a = curand_normal(state);
    float b = curand_normal(state);
    float c = curand_normal(state);
    return sqrtf(a * a + b * b + c * c);
}

__device__ int sample_fission_multiplicity(curandState *state)
{
    return random_uniform(state) < 0.5f ? 2 : 3;
}
