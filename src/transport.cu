#include "../include/sim.cuh"
#include <curand_kernel.h>
#include <math.h>

#define PI 3.14159265358979323846f

namespace {

__device__ int get_energy_group(float energy) {
  int group = 0;

#pragma unroll
  for (int threshold = 0; threshold < NUM_GROUPS - 1; ++threshold) {
    group += energy < d_GROUP_ENERGY[threshold];
  }

  return group;
}

__device__ XS get_cross_sections(float energy, int region) {
  int group = get_energy_group(energy);
  int material = region;
  if (material < 0 || material >= NUM_REGIONS) {
    material = MODERATOR;
  }

  XS cross_section;
  cross_section.sig_f = d_sigma_f[group][material];
  cross_section.sig_c = d_sigma_c[group][material];
  cross_section.sig_s = d_sigma_s[group][material];
  cross_section.sig_t =
      cross_section.sig_f + cross_section.sig_c + cross_section.sig_s;
  return cross_section;
}

__device__ bool closer_positive(float distance, float *best_distance) {
  const float eps = 1.0e-9f;
  if (distance > eps && distance < *best_distance) {
    *best_distance = distance;
    return true;
  }
  return false;
}

__device__ void check_circle_boundary(const Neutron &neutron, float radius,
                                      SurfaceId boundary_surface,
                                      float *best_distance, SurfaceId *surface) {
  float b = 2.0f * (neutron.x * neutron.ux + neutron.y * neutron.uy);
  float c = neutron.x * neutron.x + neutron.y * neutron.y - radius * radius;
  float delta = b * b - 4.0f * c;
  if (delta < 0.0f) {
    return;
  }

  float sqrt_delta = sqrtf(delta);
  if (closer_positive((-b - sqrt_delta) * 0.5f, best_distance) ||
      closer_positive((-b + sqrt_delta) * 0.5f, best_distance)) {
    *surface = boundary_surface;
  }
}

__device__ int reserve_queue_slots(int *count, int requested, int capacity,
                                   int *reserved) {
  int old_count = *count;
  while (true) {
    if (old_count >= capacity) {
      *reserved = 0;
      return capacity;
    }

    int available = capacity - old_count;
    int granted = requested < available ? requested : available;
    int observed = atomicCAS(count, old_count, old_count + granted);
    if (observed == old_count) {
      *reserved = granted;
      return old_count;
    }
    old_count = observed;
  }
}

__device__ int normalized_region(int region) {
  if (region < 0 || region >= NUM_REGIONS) {
    return MODERATOR;
  }
  return region;
}

__device__ void add_completed_region(RegionCorrectionTallies *region_correction,
                                     int region) {
  atomicAdd(&region_correction->completed[normalized_region(region)], 1ULL);
}

__device__ void add_produced_region(RegionCorrectionTallies *region_correction,
                                    int region, unsigned long long produced) {
  atomicAdd(&region_correction->produced[normalized_region(region)], produced);
}

} // namespace

__global__ void move_kernel(
    const Neutron *move_queue,
    int *move_count,
    Neutron *next_move_queue,
    int *next_move_count,
    Neutron *collision_queue,
    int *collision_count,
    Tallies *global_tallies,
    // HistoryTallies *history_tallies,
    RegionCorrectionTallies *region_correction,
    int queue_capacity,
    float r_fuel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *move_count) return;

    Neutron neutron = move_queue[i];

    float E = neutron.Energy;
    int region = neutron.region;

    XS xs = get_cross_sections(E, region);
    float sig_t = xs.sig_t; // this is the total cross section

    curandState local_state = neutron.rng_state;

    if (neutron.regionchange == 0) {
        sample_isotropic_direction(&local_state, &neutron.ux, &neutron.uy);
    } else {
        neutron.regionchange = 0;
    }

    float xi = random_uniform(&local_state);

    // sample free-flight distance (python version): d = -(1.0 / sig[3]) * np.log(np.random.random())
    // equivalent to: d = -(1.0 / sig_t) * log(a random float number between 0.0 and 1.0)
    float d = -logf(xi) / sig_t;

    float dmin = INFINITY;
    SurfaceId surface = SURFACE_NONE;
    float r_clad_out = DEFAULT_GEOMETRY.r_clad_out;
    float half_pitch = 0.5f * DEFAULT_GEOMETRY.pitch;

    if (region == FUEL) {
        check_circle_boundary(neutron, r_fuel, SURFACE_FUEL, &dmin, &surface);
    } else if (region == CLAD) {
        check_circle_boundary(neutron, r_fuel, SURFACE_FUEL, &dmin, &surface);
        check_circle_boundary(neutron, r_clad_out, SURFACE_CLAD_OUTER, &dmin, &surface);
    } else {
        check_circle_boundary(neutron, r_clad_out, SURFACE_CLAD_OUTER, &dmin, &surface);

        if (neutron.ux > 0.0f && closer_positive((half_pitch - neutron.x) / neutron.ux, &dmin)) {
            surface = SURFACE_X_MAX;
        }
        if (neutron.ux < 0.0f && closer_positive((-half_pitch - neutron.x) / neutron.ux, &dmin)) {
            surface = SURFACE_X_MIN;
        }
        if (neutron.uy > 0.0f && closer_positive((half_pitch - neutron.y) / neutron.uy, &dmin)) {
            surface = SURFACE_Y_MAX;
        }
        if (neutron.uy < 0.0f && closer_positive((-half_pitch - neutron.y) / neutron.uy, &dmin)) {
            surface = SURFACE_Y_MIN;
        }
    }

    if (surface == SURFACE_NONE) {
        atomicAdd(&global_tallies->lost_no_surface, 1ULL);
        add_completed_region(region_correction, region);
#if DEBUG_TRANSPORT
        float radius = sqrtf(neutron.x * neutron.x + neutron.y * neutron.y);
        float r_clad_out = DEFAULT_GEOMETRY.r_clad_out;
        const float region_eps = 1.0e-4f;
        bool valid_region = true;

        if (region == FUEL) {
            atomicAdd(&global_tallies->lost_no_surface_fuel, 1ULL);
            valid_region = radius <= r_fuel + region_eps;
        } else if (region == CLAD) {
            atomicAdd(&global_tallies->lost_no_surface_clad, 1ULL);
            valid_region = radius >= r_fuel - region_eps &&
                           radius <= r_clad_out + region_eps;
        } else {
            atomicAdd(&global_tallies->lost_no_surface_moderator, 1ULL);
            float half_pitch = 0.5f * DEFAULT_GEOMETRY.pitch;
            bool inside_cell = fabsf(neutron.x) <= half_pitch + region_eps &&
                               fabsf(neutron.y) <= half_pitch + region_eps;
            valid_region = radius >= r_clad_out - region_eps && inside_cell;
        }

        if (valid_region) {
            atomicAdd(&global_tallies->lost_no_surface_valid_region, 1ULL);
        } else {
            atomicAdd(&global_tallies->lost_no_surface_invalid_region, 1ULL);
        }
#endif
        return;
    }

    // HistoryTallies local_tallies = {};

    if (d >= dmin) {
        // Move particle to geometric boundary
        neutron.x += dmin * neutron.ux;
        neutron.y += dmin * neutron.uy;
        const float boundary_eps = 1.0e-4f;

        if (surface == SURFACE_FUEL) {
#if DEBUG_TRANSPORT
            if (region == FUEL) {
                atomicAdd(&global_tallies->fuel_surface_crossings, 1ULL);
            } else if (region == CLAD) {
                atomicAdd(&global_tallies->clad_surface_crossings, 1ULL);
            }
#endif
            neutron.region = region == FUEL ? CLAD : FUEL;
            neutron.regionchange = 1;
            neutron.x += boundary_eps * neutron.ux;
            neutron.y += boundary_eps * neutron.uy;
        } else if (surface == SURFACE_CLAD_OUTER) {
#if DEBUG_TRANSPORT
            if (region == CLAD) {
                atomicAdd(&global_tallies->clad_surface_crossings, 1ULL);
            }
#endif
            neutron.region = region == CLAD ? MODERATOR : CLAD;
            neutron.regionchange = 1;
            neutron.x += boundary_eps * neutron.ux;
            neutron.y += boundary_eps * neutron.uy;
        } else {
#if DEBUG_TRANSPORT
            atomicAdd(&global_tallies->square_surface_crossings, 1ULL);
#endif
            if (surface == SURFACE_X_MAX || surface == SURFACE_X_MIN) {
                // neutron.x = -neutron.x;
                neutron.ux = -neutron.ux;
            } else {
                // neutron.y = -neutron.y;
                neutron.uy = -neutron.uy;
            }
            neutron.region = MODERATOR;
            neutron.regionchange = 1;
            // atomicAdd(&global_tallies->leakage, 1ULL);
        }

        int reserved = 0;
        int idx = reserve_queue_slots(next_move_count, 1, queue_capacity, &reserved);
        if (reserved == 1) {
            neutron.rng_state = local_state;
            next_move_queue[idx] = neutron;
        } else {
            atomicAdd(&global_tallies->queue_overflow, 1ULL);
            add_completed_region(region_correction, region);
        }
    } else {
        // Move particle to collision site
        neutron.x += d * neutron.ux;
        neutron.y += d * neutron.uy;

        int reserved = 0;
        int idx = reserve_queue_slots(collision_count, 1, queue_capacity, &reserved);
        if (reserved == 1) {
            neutron.rng_state = local_state;
            collision_queue[idx] = neutron;
        } else {
            atomicAdd(&global_tallies->queue_overflow, 1ULL);
            add_completed_region(region_correction, region);
        }
    }


    // TODO: Process one movement event.
    // - get per-thread RNG state from rng.cu
    // - look up cross sections for neutron.Energy and neutron.region
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
}

__global__ void collision_kernel(const Neutron *collision_queue,
                                 const int *collision_count, Neutron *next_move_queue,
                                 int *next_move_count, Neutron *fission_bank,
                                 int fission_bank_capacity,
                                 int *fission_bank_count,
                                 HistoryTallies *history_tallies,
                                 Tallies *global_tallies,
                                 RegionCorrectionTallies *region_correction,
                                 int queue_capacity) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *collision_count)
    return;

  Neutron neutron = collision_queue[i];
  HistoryTallies local_tallies = {};
  curandState local_state = neutron.rng_state;

  XS cross_section =
      get_cross_sections(neutron.Energy, neutron.region);
  if (cross_section.sig_t <= 0.0f) {
    add_completed_region(region_correction, neutron.region);
    history_tallies[i] = local_tallies;
    return;
  }

  float reaction_sample = random_uniform(&local_state);
  float fission_probability = cross_section.sig_f / cross_section.sig_t;
  float capture_probability = cross_section.sig_c / cross_section.sig_t;

  // Warp-aggregated append:
  // 1. mark which lanes in this warp are adding particles
  // 2. count how many total output slots the warp needs
  // 3. one lane reserves all slots
  // 4. each lane writes at reserved_base + its local offset

  // Fission: lanes where fission_neutrons > 0 add children.
  if (reaction_sample <= fission_probability) {
    local_tallies.fission = 1;

    int fission_neutrons = sample_fission_multiplicity(&local_state);
    local_tallies.neutrons_produced = fission_neutrons;
    add_completed_region(region_correction, neutron.region);
    add_produced_region(region_correction, neutron.region,
                        static_cast<unsigned long long>(fission_neutrons));

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
    int reserved_items = 0;
    if (lane == leader) {
      // leader reserves spot in queue
      bank_index = reserve_queue_slots(fission_bank_count, total_items,
                                       fission_bank_capacity, &reserved_items);
      if (reserved_items < total_items) {
        atomicAdd(&global_tallies->fission_bank_overflow,
                  static_cast<unsigned long long>(total_items - reserved_items));
      }
    }

    // share queue spot with rest of warp
    bank_index = __shfl_sync(fission_lane_mask, bank_index, leader) + lane_offset;
    reserved_items = __shfl_sync(fission_lane_mask, reserved_items, leader);

    for (int child = 0; child < fission_neutrons; ++child) {
      int output_index = bank_index + child;
      if (lane_offset + child < reserved_items) {
        Neutron fission_neutron = neutron;
        sample_isotropic_direction(&local_state, &fission_neutron.ux,
                                   &fission_neutron.uy);
        fission_neutron.regionchange = 1;
        fission_neutron.rng_state = local_state;
        fission_bank[output_index] = fission_neutron;
      }
    }
  }

  // Capture: no particles added.
  else if (reaction_sample <= fission_probability + capture_probability) {
    local_tallies.capture = 1;
    add_completed_region(region_correction, neutron.region);
  }

  // Scatter: every active lane adds one neutron.
  else {
    local_tallies.scattering = 1;

    // float mass_number = 1.00794f;
    float mass_number = 4.5f;
    if (neutron.region == FUEL) {
      mass_number = 238.02891f;
    } else if (neutron.region == CLAD) {
      mass_number = 26.981539f;
    }

    // compute collisions (same as QMC)
    // float mass_ratio = (mass_number - 1.0f) / (mass_number + 1.0f);
    // float ksi = 1.0f + logf(mass_ratio) * (mass_number - 1.0f) *
    //                        (mass_number - 1.0f) / (2.0f * mass_number);
    // neutron.Energy *= expf(-ksi);

    float alpha = powf((mass_number - 1.0f) / (mass_number + 1.0f), 2.0f);
    float rand_val = random_uniform(&local_state);

    // The neutron randomly loses energy anywhere between its current Energy and alpha * Energy
    neutron.Energy *= (alpha + (1.0f - alpha) * rand_val);
    neutron.regionchange = 0;

    unsigned int active_scatter_mask = __activemask();
    unsigned int scatter_lane_mask = __ballot_sync(active_scatter_mask, true);
    int lane = threadIdx.x % 32;
    int scatter_queue_offset = __popc(scatter_lane_mask & ((1u << lane) - 1));
    int total_items = __popc(scatter_lane_mask);
    int leader = __ffs(scatter_lane_mask) - 1;

    int output_index = 0;
    int reserved_items = 0;
    if (lane == leader) {
      output_index = reserve_queue_slots(next_move_count, total_items,
                                         queue_capacity, &reserved_items);
    }

    output_index = __shfl_sync(scatter_lane_mask, output_index, leader) + scatter_queue_offset;
    reserved_items = __shfl_sync(scatter_lane_mask, reserved_items, leader);

    neutron.rng_state = local_state;
    if (lane == leader && reserved_items < total_items) {
      atomicAdd(&global_tallies->queue_overflow,
                static_cast<unsigned long long>(total_items - reserved_items));
    }
    if (scatter_queue_offset < reserved_items) {
      next_move_queue[output_index] = neutron;
    }
  }

  if (local_tallies.fission != 0) {
    atomicAdd(&global_tallies->fission,
              static_cast<unsigned long long>(local_tallies.fission));
    atomicAdd(&global_tallies->neutrons_produced,
              static_cast<unsigned long long>(local_tallies.neutrons_produced));
  }
  if (local_tallies.capture != 0) {
    atomicAdd(&global_tallies->capture,
              static_cast<unsigned long long>(local_tallies.capture));
  }
  if (local_tallies.scattering != 0) {
    atomicAdd(&global_tallies->scattering,
              static_cast<unsigned long long>(local_tallies.scattering));
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

__global__ void reset_counts_kernel(int *first_count, int *second_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *first_count = 0;
    *second_count = 0;
  }
}

__global__ void tail_correction_kernel(const Neutron *tail_queue,
                                       const int *tail_count,
                                       RegionCorrectionTallies *region_correction) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *tail_count) {
    return;
  }

  int region = normalized_region(tail_queue[i].region);
  atomicAdd(&region_correction->tail[region], 1u);
}
