#define DEBUG_TRANSPORT 0

#include "include/sim.cuh"
#include "common.h"
#include "rng.cu"
#include "fission_bank.cu"
#include <cstdio>
#include <cstdlib>

#define DEFAULT_NEUTRONS 10000
#define DEFAULT_GENERATIONS 10
#define QUEUE_MULTIPLIER 10
#define FUEL_RADIUS 0.53f
#define MAX_HISTORY_EVENTS 10000

namespace {

__device__ int history_energy_group(float energy) {
    int group = 0;

#pragma unroll
    for (int threshold = 0; threshold < NUM_GROUPS - 1; ++threshold) {
        group += energy < d_GROUP_ENERGY[threshold];
    }

    return group;
}

__device__ XS history_cross_sections(float energy, int region) {
    int group = history_energy_group(energy);
    int material = (region >= 0 && region < NUM_REGIONS) ? region : MODERATOR;
    return d_cross_sections[group][material];
}

__device__ bool history_closer_positive(float distance, float *best_distance) {
    const float eps = 1.0e-9f;
    if (distance > eps && distance < *best_distance) {
        *best_distance = distance;
        return true;
    }
    return false;
}

__device__ void history_check_circle(float x, float y, float ux, float uy,
                                     float radius, SurfaceId boundary_surface,
                                     float *best_distance, SurfaceId *surface) {
    float b = 2.0f * (x * ux + y * uy);
    float c = x * x + y * y - radius * radius;
    float delta = b * b - 4.0f * c;
    if (delta < 0.0f) {
        return;
    }

    float sqrt_delta = sqrtf(delta);
    if (history_closer_positive((-b - sqrt_delta) * 0.5f, best_distance) ||
        history_closer_positive((-b + sqrt_delta) * 0.5f, best_distance)) {
        *surface = boundary_surface;
    }
}

__device__ int history_reserve_slots(int *count, int requested, int capacity,
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

__global__ void history_kernel(
    NeutronSoA source_particles,
    int source_count,
    NeutronSoA fission_bank,
    int fission_bank_capacity,
    int *fission_bank_count,
    Tallies *global_tallies,
    int max_history_events
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= source_count) {
        return;
    }

    float x = source_particles.x[i];
    float y = source_particles.y[i];
    float energy = source_particles.Energy[i];
    float ux = source_particles.ux[i];
    float uy = source_particles.uy[i];
    int region = source_particles.region[i];
    int regionchange = source_particles.regionchange[i];
    curandState local_state = source_particles.rng_state[i];

    unsigned int local_fission = 0;
    unsigned int local_capture = 0;
    unsigned int local_scattering = 0;
    unsigned int local_neutrons_produced = 0;
    unsigned int local_lost = 0;

    for (int event = 0; event < max_history_events; ++event) {
        XS xs = history_cross_sections(energy, region);
        if (xs.sig_t <= 0.0f) {
            ++local_lost;
            break;
        }

        if (regionchange == 0) {
            sample_isotropic_direction(&local_state, &ux, &uy);
        } else {
            regionchange = 0;
        }

        float xi = random_uniform(&local_state);
        float distance_to_collision = -logf(xi) / xs.sig_t;

        float distance_to_boundary = INFINITY;
        SurfaceId surface = SURFACE_NONE;
        float r_clad_out = DEFAULT_GEOMETRY.r_clad_out;
        float half_pitch = 0.5f * DEFAULT_GEOMETRY.pitch;

        if (region == FUEL) {
            history_check_circle(x, y, ux, uy, FUEL_RADIUS, SURFACE_FUEL,
                                 &distance_to_boundary, &surface);
        } else if (region == CLAD) {
            history_check_circle(x, y, ux, uy, FUEL_RADIUS, SURFACE_FUEL,
                                 &distance_to_boundary, &surface);
            history_check_circle(x, y, ux, uy, r_clad_out, SURFACE_CLAD_OUTER,
                                 &distance_to_boundary, &surface);
        } else {
            history_check_circle(x, y, ux, uy, r_clad_out, SURFACE_CLAD_OUTER,
                                 &distance_to_boundary, &surface);

            if (ux > 0.0f && history_closer_positive((half_pitch - x) / ux,
                                                     &distance_to_boundary)) {
                surface = SURFACE_X_MAX;
            }
            if (ux < 0.0f && history_closer_positive((-half_pitch - x) / ux,
                                                     &distance_to_boundary)) {
                surface = SURFACE_X_MIN;
            }
            if (uy > 0.0f && history_closer_positive((half_pitch - y) / uy,
                                                     &distance_to_boundary)) {
                surface = SURFACE_Y_MAX;
            }
            if (uy < 0.0f && history_closer_positive((-half_pitch - y) / uy,
                                                     &distance_to_boundary)) {
                surface = SURFACE_Y_MIN;
            }
        }

        if (surface == SURFACE_NONE) {
            ++local_lost;
            break;
        }

        if (distance_to_collision >= distance_to_boundary) {
            x += distance_to_boundary * ux;
            y += distance_to_boundary * uy;

            if (surface == SURFACE_FUEL) {
                region = region == FUEL ? CLAD : FUEL;
                regionchange = 1;
                const float boundary_eps = 1.0e-4f;
                x += boundary_eps * ux;
                y += boundary_eps * uy;
            } else if (surface == SURFACE_CLAD_OUTER) {
                region = region == CLAD ? MODERATOR : CLAD;
                regionchange = 1;
                const float boundary_eps = 1.0e-4f;
                x += boundary_eps * ux;
                y += boundary_eps * uy;
            } else {
                if (surface == SURFACE_X_MAX || surface == SURFACE_X_MIN) {
                    ux = -ux;
                } else {
                    uy = -uy;
                }
                region = MODERATOR;
                regionchange = 1;
            }

            continue;
        }

        x += distance_to_collision * ux;
        y += distance_to_collision * uy;

        float reaction_sample = random_uniform(&local_state);
        float fission_probability = xs.sig_f / xs.sig_t;
        float capture_probability = xs.sig_c / xs.sig_t;

        if (reaction_sample <= fission_probability) {
            ++local_fission;
            int fission_neutrons = sample_fission_multiplicity(&local_state);
            local_neutrons_produced += static_cast<unsigned int>(fission_neutrons);

            int reserved = 0;
            int base = history_reserve_slots(fission_bank_count, fission_neutrons,
                                             fission_bank_capacity, &reserved);
            if (reserved < fission_neutrons) {
                atomicAdd(&global_tallies->fission_bank_overflow,
                          static_cast<unsigned long long>(fission_neutrons - reserved));
            }

            for (int child = 0; child < reserved; ++child) {
                int output_index = base + child;
                float child_ux;
                float child_uy;
                sample_isotropic_direction(&local_state, &child_ux, &child_uy);
                fission_bank.x[output_index] = x;
                fission_bank.y[output_index] = y;
                fission_bank.Energy[output_index] = energy;
                fission_bank.ux[output_index] = child_ux;
                fission_bank.uy[output_index] = child_uy;
                fission_bank.region[output_index] = region;
                fission_bank.regionchange[output_index] = 1;
                fission_bank.rng_state[output_index] = local_state;
            }
            break;
        }

        if (reaction_sample <= fission_probability + capture_probability) {
            ++local_capture;
            break;
        }

        ++local_scattering;
        float mass_number = 4.5f;
        if (region == FUEL) {
            mass_number = 238.02891f;
        } else if (region == CLAD) {
            mass_number = 26.981539f;
        }

        float alpha = powf((mass_number - 1.0f) / (mass_number + 1.0f), 2.0f);
        energy *= alpha + (1.0f - alpha) * random_uniform(&local_state);
        regionchange = 0;
    }

    source_particles.rng_state[i] = local_state;

    if (local_fission != 0) {
        atomicAdd(&global_tallies->fission,
                  static_cast<unsigned long long>(local_fission));
        atomicAdd(&global_tallies->neutrons_produced,
                  static_cast<unsigned long long>(local_neutrons_produced));
    }
    if (local_capture != 0) {
        atomicAdd(&global_tallies->capture,
                  static_cast<unsigned long long>(local_capture));
    }
    if (local_scattering != 0) {
        atomicAdd(&global_tallies->scattering,
                  static_cast<unsigned long long>(local_scattering));
    }
    if (local_lost != 0) {
        atomicAdd(&global_tallies->lost_no_surface,
                  static_cast<unsigned long long>(local_lost));
    }
}

int main(int argc, char **argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : DEFAULT_NEUTRONS;
    int num_generations = argc > 2 ? std::atoi(argv[2]) : DEFAULT_GENERATIONS;

    if (N <= 0 || num_generations <= 0 || argc > 3) {
        printf("Usage: %s [neutrons] [generations]\n", argv[0]);
        return 1;
    }

    int queue_capacity = N * QUEUE_MULTIPLIER;
    int fission_bank_capacity = queue_capacity;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int capacity_blocks = (queue_capacity + threads - 1) / threads;

    printf("History GPU Config: neutrons=%d generations=%d queue_capacity=%d max_history_events=%d\n",
           N, num_generations, queue_capacity, MAX_HISTORY_EVENTS);

    init_cross_sections();

    curandState *d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, queue_capacity * sizeof(curandState)));
    init_rng<<<capacity_blocks, threads>>>(d_rng_states, 1234UL, queue_capacity);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto alloc_soa = [](NeutronSoA &soa, int capacity) {
        CUDA_CHECK(cudaMalloc(&soa.x, capacity * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&soa.y, capacity * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&soa.Energy, capacity * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&soa.ux, capacity * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&soa.uy, capacity * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&soa.region, capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&soa.regionchange, capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&soa.rng_state, capacity * sizeof(curandState)));
    };

    auto free_soa = [](NeutronSoA &soa) {
        CUDA_CHECK(cudaFree(soa.x));
        CUDA_CHECK(cudaFree(soa.y));
        CUDA_CHECK(cudaFree(soa.Energy));
        CUDA_CHECK(cudaFree(soa.ux));
        CUDA_CHECK(cudaFree(soa.uy));
        CUDA_CHECK(cudaFree(soa.region));
        CUDA_CHECK(cudaFree(soa.regionchange));
        CUDA_CHECK(cudaFree(soa.rng_state));
    };

    NeutronSoA d_source_particles;
    NeutronSoA d_fission_bank;
    alloc_soa(d_source_particles, queue_capacity);
    alloc_soa(d_fission_bank, fission_bank_capacity);

    Tallies *d_global_tallies;
    int *d_fission_bank_count;
    CUDA_CHECK(cudaMalloc(&d_global_tallies, sizeof(Tallies)));
    CUDA_CHECK(cudaMemset(d_global_tallies, 0, sizeof(Tallies)));
    CUDA_CHECK(cudaMalloc(&d_fission_bank_count, sizeof(int)));

    initialize_neutrons<<<blocks, threads>>>(d_rng_states, d_source_particles, FUEL_RADIUS, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int completed_generations = 0;
    double keff_sum = 0.0;
    int fission_bank_count = 0;

    for (int generation = 0; generation < num_generations; ++generation) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_fission_bank_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

        history_kernel<<<blocks, threads>>>(
            d_source_particles,
            N,
            d_fission_bank,
            fission_bank_capacity,
            d_fission_bank_count,
            d_global_tallies,
            MAX_HISTORY_EVENTS
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&fission_bank_count, d_fission_bank_count,
                              sizeof(int), cudaMemcpyDeviceToHost));
        double generation_keff = static_cast<double>(fission_bank_count) /
            static_cast<double>(N);
        printf("Generation %d Fission Bank Sites.........=  %d\n",
               generation, fission_bank_count);
        printf("Generation %d keff estimate..............=  %.12f\n",
               generation, generation_keff);
        keff_sum += generation_keff;
        completed_generations = generation + 1;

        if (generation + 1 >= num_generations || fission_bank_count <= 0) {
            break;
        }

        resample_kernel<<<blocks, threads>>>(
            d_fission_bank,
            fission_bank_count,
            d_source_particles,
            N,
            d_rng_states
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    Tallies global_tallies = {};
    CUDA_CHECK(cudaMemcpy(&global_tallies, d_global_tallies,
                          sizeof(Tallies), cudaMemcpyDeviceToHost));

    unsigned long long interactions =
        global_tallies.scattering + global_tallies.capture + global_tallies.fission;
    unsigned long long absorption = global_tallies.capture + global_tallies.fission;
    double average_nu = global_tallies.fission > 0
        ? static_cast<double>(global_tallies.neutrons_produced) /
              static_cast<double>(global_tallies.fission)
        : 0.0;

    printf("Number of Neutrons.......................=  %d\n", N);
    printf("Completed Generations....................=  %d\n", completed_generations);
    if (completed_generations > 0) {
        printf("Average keff.............................=  %.12f\n",
               keff_sum / static_cast<double>(completed_generations));
    }
    printf("Number of Interactions...................=  %llu\n", interactions);
    printf("Number of Scattering Events..............=  %llu\n", global_tallies.scattering);
    printf("Number of Capture Events.................=  %llu\n", global_tallies.capture);
    printf("Number of Fission Events.................=  %llu\n", global_tallies.fission);
    printf("Number of Absorption Events..............=  %llu\n", absorption);
    printf("Average nu...............................=  %.6f\n", average_nu);
    printf("Number of Neutrons Produced by Fission...=  %llu\n", global_tallies.neutrons_produced);
    printf("Number of Fission Bank Sites.............=  %d\n", fission_bank_count);
    printf("Number of Neutrons Leaked from System....=  %llu\n", global_tallies.leakage);
    printf("Fission Bank Overflowed Particles........=  %llu\n", global_tallies.fission_bank_overflow);
    printf("Lost With No Surface.....................=  %llu\n", global_tallies.lost_no_surface);

    CUDA_CHECK(cudaFree(d_fission_bank_count));
    CUDA_CHECK(cudaFree(d_global_tallies));
    free_soa(d_fission_bank);
    free_soa(d_source_particles);
    CUDA_CHECK(cudaFree(d_rng_states));

    printf("history simulation ended!\n");
    return 0;
}
