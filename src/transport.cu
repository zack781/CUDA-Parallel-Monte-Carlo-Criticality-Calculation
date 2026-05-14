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
  int material = (region >= 0 && region < NUM_REGIONS) ? region : MODERATOR;
  return d_cross_sections[group][material];
}

__device__ bool closer_positive(float distance, float *best_distance) {
  const float eps = 1.0e-9f;
  if (distance > eps && distance < *best_distance) {
    *best_distance = distance;
    return true;
  }
  return false;
}

__device__ void check_circle_boundary(float x, float y, float ux, float uy,
                                      float radius, SurfaceId boundary_surface,
                                      float *best_distance, SurfaceId *surface) {
  float b = 2.0f * (x * ux + y * uy);
  float c = x * x + y * y - radius * radius;
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

} // namespace

__global__ void move_kernel(
    NeutronSoA move_queue,
    int *move_count,
    NeutronSoA next_move_queue,
    int *next_move_count,
    NeutronSoA collision_queue,
    int *collision_count,
    Tallies *global_tallies,
    int queue_capacity,
    float r_fuel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *move_count) return;

    float x           = move_queue.x[i];
    float y           = move_queue.y[i];
    float E           = move_queue.Energy[i];
    float ux          = move_queue.ux[i];
    float uy          = move_queue.uy[i];
    int region        = move_queue.region[i];
    int regionchange  = move_queue.regionchange[i];
    curandState local_state = move_queue.rng_state[i];

    XS xs = get_cross_sections(E, region);
    float sig_t = xs.sig_t;

    if (regionchange == 0) {
        sample_isotropic_direction(&local_state, &ux, &uy);
    } else {
        regionchange = 0;
    }

    float xi = random_uniform(&local_state);
    float d = -logf(xi) / sig_t;

    float dmin = INFINITY;
    SurfaceId surface = SURFACE_NONE;
    float r_clad_out = DEFAULT_GEOMETRY.r_clad_out;
    float half_pitch = 0.5f * DEFAULT_GEOMETRY.pitch;

    if (region == FUEL) {
        check_circle_boundary(x, y, ux, uy, r_fuel, SURFACE_FUEL, &dmin, &surface);
    } else if (region == CLAD) {
        check_circle_boundary(x, y, ux, uy, r_fuel, SURFACE_FUEL, &dmin, &surface);
        check_circle_boundary(x, y, ux, uy, r_clad_out, SURFACE_CLAD_OUTER, &dmin, &surface);
    } else {
        check_circle_boundary(x, y, ux, uy, r_clad_out, SURFACE_CLAD_OUTER, &dmin, &surface);

        if (ux > 0.0f && closer_positive((half_pitch - x) / ux, &dmin)) {
            surface = SURFACE_X_MAX;
        }
        if (ux < 0.0f && closer_positive((-half_pitch - x) / ux, &dmin)) {
            surface = SURFACE_X_MIN;
        }
        if (uy > 0.0f && closer_positive((half_pitch - y) / uy, &dmin)) {
            surface = SURFACE_Y_MAX;
        }
        if (uy < 0.0f && closer_positive((-half_pitch - y) / uy, &dmin)) {
            surface = SURFACE_Y_MIN;
        }
    }

    if (surface == SURFACE_NONE) {
        atomicAdd(&global_tallies->lost_no_surface, 1ULL);
#if DEBUG_TRANSPORT
        float radius = sqrtf(x * x + y * y);
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
            bool inside_cell = fabsf(x) <= half_pitch + region_eps &&
                               fabsf(y) <= half_pitch + region_eps;
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

    if (d >= dmin) {
        x += dmin * ux;
        y += dmin * uy;
        const float boundary_eps = 1.0e-4f;

        if (surface == SURFACE_FUEL) {
#if DEBUG_TRANSPORT
            if (region == FUEL) {
                atomicAdd(&global_tallies->fuel_surface_crossings, 1ULL);
            } else if (region == CLAD) {
                atomicAdd(&global_tallies->clad_surface_crossings, 1ULL);
            }
#endif
            region = (region == FUEL) ? CLAD : FUEL;
            regionchange = 1;
            x += boundary_eps * ux;
            y += boundary_eps * uy;
        } else if (surface == SURFACE_CLAD_OUTER) {
#if DEBUG_TRANSPORT
            if (region == CLAD) {
                atomicAdd(&global_tallies->clad_surface_crossings, 1ULL);
            }
#endif
            region = (region == CLAD) ? MODERATOR : CLAD;
            regionchange = 1;
            x += boundary_eps * ux;
            y += boundary_eps * uy;
        } else {
#if DEBUG_TRANSPORT
            atomicAdd(&global_tallies->square_surface_crossings, 1ULL);
#endif
            if (surface == SURFACE_X_MAX || surface == SURFACE_X_MIN) {
                ux = -ux;
            } else {
                uy = -uy;
            }
            region = MODERATOR;
            regionchange = 1;
        }

        int reserved = 0;
        int idx = reserve_queue_slots(next_move_count, 1, queue_capacity, &reserved);
        if (reserved == 1) {
            next_move_queue.x[idx]           = x;
            next_move_queue.y[idx]           = y;
            next_move_queue.Energy[idx]      = E;
            next_move_queue.ux[idx]          = ux;
            next_move_queue.uy[idx]          = uy;
            next_move_queue.region[idx]      = region;
            next_move_queue.regionchange[idx] = regionchange;
            next_move_queue.rng_state[idx]   = local_state;
        } else {
            atomicAdd(&global_tallies->queue_overflow, 1ULL);
        }
    } else {
        x += d * ux;
        y += d * uy;

        int reserved = 0;
        int idx = reserve_queue_slots(collision_count, 1, queue_capacity, &reserved);
        if (reserved == 1) {
            collision_queue.x[idx]           = x;
            collision_queue.y[idx]           = y;
            collision_queue.Energy[idx]      = E;
            collision_queue.ux[idx]          = ux;
            collision_queue.uy[idx]          = uy;
            collision_queue.region[idx]      = region;
            collision_queue.regionchange[idx] = regionchange;
            collision_queue.rng_state[idx]   = local_state;
        } else {
            atomicAdd(&global_tallies->queue_overflow, 1ULL);
        }
    }
}

__global__ void collision_kernel(NeutronSoA collision_queue,
                                 const int *collision_count, NeutronSoA next_move_queue,
                                 int *next_move_count, NeutronSoA fission_bank,
                                 int fission_bank_capacity,
                                 int *fission_bank_count,
                                 HistoryTallies *history_tallies,
                                 Tallies *global_tallies,
                                 int queue_capacity) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *collision_count)
    return;

  float x          = collision_queue.x[i];
  float y          = collision_queue.y[i];
  float Energy     = collision_queue.Energy[i];
  float ux         = collision_queue.ux[i];
  float uy         = collision_queue.uy[i];
  int region       = collision_queue.region[i];
  int regionchange = collision_queue.regionchange[i];
  curandState local_state = collision_queue.rng_state[i];

  HistoryTallies local_tallies = {};

  XS cross_section = get_cross_sections(Energy, region);
  if (cross_section.sig_t <= 0.0f) {
    history_tallies[i] = local_tallies;
    return;
  }

  float reaction_sample = random_uniform(&local_state);
  float fission_probability = cross_section.sig_f / cross_section.sig_t;
  float capture_probability = cross_section.sig_c / cross_section.sig_t;

  // Fission: lanes where fission_neutrons > 0 add children.
  if (reaction_sample <= fission_probability) {
    local_tallies.fission = 1;

    int fission_neutrons = sample_fission_multiplicity(&local_state);
    local_tallies.neutrons_produced = fission_neutrons;

    unsigned int active_fission_mask = __activemask();
    unsigned int fission_lane_mask =
        __ballot_sync(active_fission_mask, fission_neutrons > 0);
    int lane = threadIdx.x & 31;
    int lane_offset = 0;
    int total_items = 0;

#pragma unroll
    for (int source_lane = 0; source_lane < 32; ++source_lane) {
      unsigned int source_bit = 1u << source_lane;
      if ((fission_lane_mask & source_bit) != 0) {
        int source_items =
            __shfl_sync(fission_lane_mask, fission_neutrons, source_lane);
        total_items += source_items;
        if (source_lane < lane) {
          lane_offset += source_items;
        }
      }
    }

    int leader = __ffs(fission_lane_mask) - 1;
    int bank_index = 0;
    int reserved_items = 0;
    if (lane == leader) {
      bank_index = reserve_queue_slots(fission_bank_count, total_items,
                                       fission_bank_capacity, &reserved_items);
      if (reserved_items < total_items) {
        atomicAdd(&global_tallies->fission_bank_overflow,
                  static_cast<unsigned long long>(total_items - reserved_items));
      }
    }

    bank_index = __shfl_sync(fission_lane_mask, bank_index, leader) + lane_offset;
    reserved_items = __shfl_sync(fission_lane_mask, reserved_items, leader);

    for (int child = 0; child < fission_neutrons; ++child) {
      int output_index = bank_index + child;
      if (lane_offset + child < reserved_items) {
        float f_ux, f_uy;
        sample_isotropic_direction(&local_state, &f_ux, &f_uy);
        fission_bank.x[output_index]            = x;
        fission_bank.y[output_index]            = y;
        fission_bank.Energy[output_index]       = Energy;
        fission_bank.ux[output_index]           = f_ux;
        fission_bank.uy[output_index]           = f_uy;
        fission_bank.region[output_index]       = region;
        fission_bank.regionchange[output_index] = 1;
        fission_bank.rng_state[output_index]    = local_state;
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

    float mass_number = 4.5f;
    if (region == FUEL) {
      mass_number = 238.02891f;
    } else if (region == CLAD) {
      mass_number = 26.981539f;
    }

    float alpha = powf((mass_number - 1.0f) / (mass_number + 1.0f), 2.0f);
    float rand_val = random_uniform(&local_state);
    Energy *= (alpha + (1.0f - alpha) * rand_val);
    regionchange = 0;

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

    if (lane == leader && reserved_items < total_items) {
      atomicAdd(&global_tallies->queue_overflow,
                static_cast<unsigned long long>(total_items - reserved_items));
    }
    if (scatter_queue_offset < reserved_items) {
      next_move_queue.x[output_index]            = x;
      next_move_queue.y[output_index]            = y;
      next_move_queue.Energy[output_index]       = Energy;
      next_move_queue.ux[output_index]           = ux;
      next_move_queue.uy[output_index]           = uy;
      next_move_queue.region[output_index]       = region;
      next_move_queue.regionchange[output_index] = regionchange;
      next_move_queue.rng_state[output_index]    = local_state;
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

__global__ void compact_queue_kernel(NeutronSoA input_queue,
                                     const int *keep_flags,
                                     const int *output_offsets, int input_count,
                                     NeutronSoA output_queue) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= input_count) {
    return;
  }

  if (keep_flags[i] != 0) {
    int o = output_offsets[i];
    output_queue.x[o]            = input_queue.x[i];
    output_queue.y[o]            = input_queue.y[i];
    output_queue.Energy[o]       = input_queue.Energy[i];
    output_queue.ux[o]           = input_queue.ux[i];
    output_queue.uy[o]           = input_queue.uy[i];
    output_queue.region[o]       = input_queue.region[i];
    output_queue.regionchange[o] = input_queue.regionchange[i];
    output_queue.rng_state[o]    = input_queue.rng_state[i];
  }
}

__global__ void reset_counts_kernel(int *first_count, int *second_count) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *first_count = 0;
    *second_count = 0;
  }
}

__global__ void gather_kernel(NeutronSoA src, const int *indices, int count, NeutronSoA dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int j = indices[i];
  dst.x[i]            = src.x[j];
  dst.y[i]            = src.y[j];
  dst.Energy[i]       = src.Energy[j];
  dst.ux[i]           = src.ux[j];
  dst.uy[i]           = src.uy[j];
  dst.region[i]       = src.region[j];
  dst.regionchange[i] = src.regionchange[j];
  dst.rng_state[i]    = src.rng_state[j];
}
