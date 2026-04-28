#include "../include/sim.cuh"

namespace {

__device__ int get_energy_group(float energy) {
  int group = 0;

#pragma unroll
  for (int threshold = 1; threshold < NUM_GROUPS; ++threshold) {
    group += energy < GROUP_ENERGY[threshold];
  }

  return group;
}

__device__ CrossSections get_cross_sections(float energy, int region) {
  int group = get_energy_group(energy);
  int material = region;
  if (material < 0 || material >= NUM_REGIONS) {
    material = MODERATOR;
  }

  CrossSections cross_section;
  cross_section.fission = SIGMA_F[group][material];
  cross_section.capture = SIGMA_C[group][material];
  cross_section.scattering = SIGMA_S[group][material];
  cross_section.total =
      cross_section.fission + cross_section.capture + cross_section.scattering;
  return cross_section;
}

} // namespace

__global__ void move_kernel(
    const Neutron *move_queue,
    int move_count,
    Neutron *next_move_queue,
    int *next_move_count,
    Neutron *collision_queue,
    int *collision_count,
    // HistoryTallies *history_tallies,
    curandState *rng_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= move_count) return;

    Neutron neutron = move_queue[i];

    float E = neutron.Energy;
    int region = neutron.region;

    XS xs = CrossSections(E, region);

    float sig_f = xs.sig_f;
    float sig_c = xs.sig_c;
    float sig_s = xs.sig_s;
    float sig_t = xs.sig_t; // this is the total cross section

    curandState local_state = rng_states[i];
    float xi = curand_uniform(&local_state);

    // sample free-flight distance (python version): d = -(1.0 / sig[3]) * np.log(np.random.random())
    // equivalent to: d = -(1.0 / sig_t) * log(a random float number between 0.0 and 1.0)
    float d = -logf(xi) / sig_t;

    // distance to geometric boundary
    // define theta
    float theta = 2 * PI * curand_uniform(&local_state);
    float ux = cosf(theta);
    float uy = sinf(theta);

    float a = 1.0;
    float b = 2.0f * (neutron.x * ux + neutron.y * uy);
    float x = neutron.x * neutron.x + neutron.y * neutron.y - r_fuel * r_fuel;
    float delta = b * b - 4.0f * a * c;

    float dmin = INFINITY;

    if (delta >= 0.0f) {
        float sqrt_delta = sqrtf(delta);

        float df1 = (-b - sqrt_delta) / (2.0f * a);
        float df2 = (-b + sqrt_delta) / (2.0f * a);

        const float eps = 1e-9f;

        if (df1 > eps && df2 > eps) {
            dmin = fminf(df1, df2);
        } else if (df1 > eps) {
            dmin = df1;
        } else if (df2 > eps) {
            dmin = df2;
        } else {
            rng_states[i] = local_state;
            return;  // no forward boundary intersection
        }
    } else {
        rng_states[i] = local_state;
        return;      // no boundary intersection
    }

    // HistoryTallies local_tallies = {};

    if (d >= dmin) {
        // Move particle to geometric boundary
        neutron.x += dmin * ux;
        neutron.y += dmin * uy;
        neutron.region = 1;

        int idx = atomicAdd(next_move_count, 1);
        next_move_queue[idx] = neutron;
    } else {
        // Move particle to collision site
        neutron.x += d * ux;
        neutron.y += d * uy;

        int idx = atomicAdd(collision_count, 1);
        collision_queue[idx] = neutron;
    }


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

    // history_tallies[i] = local_tallies;

    rng_states[i] = local_state; // prevents next curand call from generating the same random number
}

__global__ void collision_kernel(const Neutron *collision_queue,
                                 int collision_count, Neutron *move_queue,
                                 int *move_count, Neutron *fission_bank,
                                 int fission_bank_capacity,
                                 int *fission_bank_count,
                                 HistoryTallies *history_tallies) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= collision_count)
    return;

  Neutron neutron = collision_queue[i];
  HistoryTallies local_tallies = {};
  unsigned int rng_state = 0u;

  CrossSections cross_section =
      get_cross_sections(neutron.energy, neutron.region);
  if (cross_section.total <= 0.0f) {
    history_tallies[i] = local_tallies;
    return;
  }

  float reaction_sample = random_uniform(&rng_state);
  float fission_probability = cross_section.fission / cross_section.total;
  float capture_probability = cross_section.capture / cross_section.total;

  // Warp-aggregated append:
  // 1. mark which lanes in this warp are adding particles
  // 2. count how many total output slots the warp needs
  // 3. one lane reserves all slots
  // 4. each lane writes at reserved_base + its local offset

  // Fission: lanes where fission_neutrons > 0 add children.
  if (reaction_sample <= fission_probability) {
    local_tallies.fission = 1;

    int fission_neutrons = random_uniform(&rng_state) < 0.5f ? 2 : 3;
    local_tallies.neutrons_produced = fission_neutrons;

    unsigned int active_fission_mask = __activemask();
    // mask of lanes where fission_neutrons > 0 (fission lanes)
    unsigned int fission_lane_mask =
        __ballot_sync(active_fission_mask, fission_neutrons > 0);
    int lane = threadIdx.x & 31;
    int lane_offset = 0;
    int total_items = 0;

#pragma unroll
    for (int source_lane = 0; source_lane < 32; ++source_lane) {
      unsigned int source_bit = 1u << source_lane;
      if ((fission_lane_mask & source_bit) != 0) {
        // share total number of fission neutrons within warps
        int source_items =
            __shfl_sync(fission_lane_mask, fission_neutrons, source_lane);
        total_items += source_items;
        if (source_lane < lane) {
          lane_offset += source_items;
        }
      }
    }

    // get the first lane that has to write
    int leader = __ffs(fission_lane_mask) - 1;
    int bank_index = 0;
    if (lane == leader) {
      // leader reserves spot in queue
      bank_index = atomicAdd(fission_bank_count, total_items);
    }

    // share queue spot with rest of warp
    bank_index =
        __shfl_sync(fission_lane_mask, bank_index, leader) + lane_offset;

    for (int child = 0; child < fission_neutrons; ++child) {
      int output_index = bank_index + child;
      if (output_index < fission_bank_capacity) {
        Neutron fission_neutron = neutron;
        sample_isotropic_direction(&rng_state, &fission_neutron.ux,
                                   &fission_neutron.uy);
        fission_bank[output_index] = fission_neutron;
      }
    }
  }

  // Capture: no particles added.
  else if (reaction_sample <= fission_probability + capture_probability) {
    local_tallies.capture = 1;
  }

  // Scatter: every active lane adds one neutron.
  else {
    local_tallies.scattering = 1;

    float mass_number = 1.00794f;
    if (neutron.region == FUEL) {
      mass_number = 238.02891f;
    } else if (neutron.region == CLAD) {
      mass_number = 26.981539f;
    }

    // compute collisions (same as QMC)
    float mass_ratio = (mass_number - 1.0f) / (mass_number + 1.0f);
    float ksi = 1.0f + logf(mass_ratio) * (mass_number - 1.0f) *
                           (mass_number - 1.0f) / (2.0f * mass_number);
    neutron.energy *= expf(-ksi);
    sample_isotropic_direction(&rng_state, &neutron.ux, &neutron.uy);

    unsigned int active_scatter_mask = __activemask();
    unsigned int scatter_lane_mask = __ballot_sync(active_scatter_mask, true);
    int lane = threadIdx.x % 32;
    int scatter_queue_offset = __popc(scatter_lane_mask & ((1u << lane) - 1));
    int total_items = __popc(scatter_lane_mask);
    int leader = __ffs(scatter_lane_mask) - 1;

    int output_index = 0;
    if (lane == leader) {
      output_index = atomicAdd(move_count, total_items);
    }

    output_index = __shfl_sync(scatter_lane_mask, output_index, leader) + scatter_queue_offset;

    move_queue[output_index] = neutron;
  }

  history_tallies[i] = local_tallies;
}

__global__ void compact_queue_kernel(const Neutron *input_queue,
                                     const int *keep_flags,
                                     const int *output_offsets, int input_count,
                                     Neutron *output_queue) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= input_count) {
    return;
  }

  if (keep_flags[i] != 0) {
    int output_index = output_offsets[i];
    output_queue[output_index] = input_queue[i];
  }
}
