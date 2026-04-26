#include "../include/sim.cuh"

namespace {

__device__ int get_energy_group(float energy)
{
    int group = 0;

    #pragma unroll
    for (int threshold = 1; threshold < NUM_GROUPS; ++threshold) {
        group += energy < GROUP_ENERGY[threshold];
    }

    return group;
}

__device__ CrossSections get_cross_sections(float energy, int region)
{
    int group = get_energy_group(energy);
    int material = region;
    if (material < 0 || material >= NUM_REGIONS) {
        material = MODERATOR;
    }

    CrossSections cross_section;
    cross_section.fission = SIGMA_F[group][material];
    cross_section.capture = SIGMA_C[group][material];
    cross_section.scattering = SIGMA_S[group][material];
    cross_section.total = cross_section.fission + cross_section.capture + cross_section.scattering;
    return cross_section;
}

}

__global__ void move_kernel(
    const Neutron *move_queue,
    int move_count,
    Neutron *next_move_queue,
    int *next_move_count,
    Neutron *collision_queue,
    int *collision_count,
    HistoryTallies *history_tallies
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= move_count) return;

    Neutron neutron = move_queue[i];
    HistoryTallies local_tallies = {};

    // TODO: Process one movement event.
    // - get per-thread RNG state from rng.cu
    // - look up cross sections for neutron.energy and neutron.region
    // - compute distance to nearest boundary
    // - compute fuel/clad/square boundary candidates
    // - mask invalid boundaries by region
    // - pick nearest valid boundary
    // - compute distance to next collision
    // - move neutron to whichever event is closer
    // - if boundary is closer:
    // - update cell/material
    // - update fuel/clad surface tallies in local_tallies
    // - apply periodic boundary wrapping and leakage tally if needed
    // - push neutron to next_move_queue
    // - if collision is closer:
    // - push neutron to collision_queue
    // - collision_kernel will sample scatter/capture/fission

    history_tallies[i] = local_tallies;
}

__global__ void collision_kernel(
    const Neutron *collision_queue,
    int collision_count,
    Neutron *move_queue,
    int *move_count,
    Neutron *fission_bank,
    int fission_bank_capacity,
    int *fission_bank_count,
    HistoryTallies *history_tallies
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= collision_count) return;

    Neutron neutron = collision_queue[i];
    HistoryTallies local_tallies = {};

    // TODO: Process one collision event.
    // - get per-thread RNG state from rng.cu
    // - look up cross sections for neutron.energy and neutron.region
    // - sample reaction from fission/capture/scattering probabilities
    // - if scatter:
    // - update neutron energy
    // - sample new direction
    // - update local_tallies.scattering
    // - push neutron back to move_queue
    // - if capture:
    // - update local_tallies.capture
    // - kill particle by not pushing it to any queue
    // - if fission:
    // - update local_tallies.fission and local_tallies.neutrons_produced
    // - write children to fission_bank
    // - kill parent by not pushing it to any queue

    history_tallies[i] = local_tallies;
}

__global__ void compact_queue_kernel(
    const Neutron *input_queue,
    const int *keep_flags,
    const int *output_offsets,
    int input_count,
    Neutron *output_queue
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= input_count) return;

    // TODO: Rebuild a dense queue after event processing.
    // - remove particles that were killed, leaked, or moved to another queue
    // - keep only particles that should continue in this queue
    // - pack surviving particles contiguously into output_queue
    //
    // - keep_flags[i] is 1 if input_queue[i] should survive, 0 otherwise
    // - output_offsets comes from an exclusive prefix sum of keep_flags
    // - if keep_flags[i] is 1, write input_queue[i] to output_queue[output_offsets[i]]
    // - output queue length is output_offsets[input_count - 1] + keep_flags[input_count - 1]
    if (keep_flags[i]) {
        output_queue[output_offsets[i]] = input_queue[i];
    }
}
