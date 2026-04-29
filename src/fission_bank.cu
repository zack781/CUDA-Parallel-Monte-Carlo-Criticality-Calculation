#include "../include/sim.cuh"

__global__ void resample_kernel(
    const Neutron *fission_bank,
    int fission_bank_count,
    Neutron *source_particles,
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

    Neutron neutron = fission_bank[bank_index];
    neutron.Energy = sample_initial_energy(&local_state);
    neutron.ux = 1.0f;
    neutron.uy = 0.0f;
    neutron.region = FUEL;
    neutron.regionchange = 0;
    neutron.rng_state = local_state;

    source_particles[idx] = neutron;
    rng_states[idx] = local_state;
}
