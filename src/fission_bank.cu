#include "../include/sim.cuh"

__global__ void resample_kernel(
    NeutronSoA fission_bank,
    int fission_bank_count,
    NeutronSoA source_particles,
    int source_count,
    curandState *rng_states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= source_count || fission_bank_count <= 0) {
        return;
    }

    curandState local_state = rng_states[idx];
    int bank_index = static_cast<int>(random_uniform(&local_state) * fission_bank_count);
    if (bank_index >= fission_bank_count) {
        bank_index = fission_bank_count - 1;
    }

    source_particles.x[idx]            = fission_bank.x[bank_index];
    source_particles.y[idx]            = fission_bank.y[bank_index];
    source_particles.Energy[idx]       = sample_initial_energy(&local_state);
    source_particles.ux[idx]           = 1.0f;
    source_particles.uy[idx]           = 0.0f;
    source_particles.region[idx]       = FUEL;
    source_particles.regionchange[idx] = 0;
    source_particles.rng_state[idx]    = local_state;

    rng_states[idx] = local_state;
}
