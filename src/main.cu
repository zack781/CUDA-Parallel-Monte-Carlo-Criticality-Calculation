#define DEBUG_TRANSPORT 0

#include "include/sim.cuh"
#include "common.h"
#include "transport.cu"
#include "rng.cu"
#include "fission_bank.cu"
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

#define DEFAULT_NEUTRONS 10000
#define DEFAULT_GENERATIONS 10
#define QUEUE_MULTIPLIER 10
#define FUEL_RADIUS 0.53
#define SORT_INTERVAL 5
#define CPU_SWITCH_THRESHOLD 4096

// move_queue: contains neutrons that need to be moved to their next event
// next_move_queue: contains neutrons that hit a geometric boundary and need transport in the next iteration
// collision_queue: contains neutrons that reached their collision site

namespace {

constexpr float CPU_PI = 3.14159265358979323846f;

struct CpuRng {
    std::mt19937 eng;
    std::uniform_real_distribution<float> uniform_dist{0.0f, 1.0f};

    explicit CpuRng(uint64_t seed) : eng(seed) {}

    float uniform() {
        return uniform_dist(eng);
    }

    void isotropic_dir(float &ux, float &uy) {
        float theta = 2.0f * CPU_PI * uniform();
        ux = std::cos(theta);
        uy = std::sin(theta);
    }

    int fission_nu() {
        return uniform() < 0.5f ? 2 : 3;
    }
};

struct CpuFissionSite {
    float x;
    float y;
};

struct CpuParticle {
    float x;
    float y;
    float Energy;
    float ux;
    float uy;
    int region;
    int regionchange;
};

int host_energy_group(float energy) {
    static const float group_energy[NUM_GROUPS] = {
        3.0e+1f, 3.0e+0f, 3.0e-1f, 3.0e-2f, 3.0e-3f,
        3.0e-4f, 3.0e-5f, 3.0e-6f, 3.0e-7f, 3.0e-8f
    };

    int group = 0;
    for (int threshold = 0; threshold < NUM_GROUPS - 1; ++threshold) {
        group += energy < group_energy[threshold];
    }
    return group;
}

XS host_cross_sections(float energy, int region) {
    static const float sigma_f[NUM_GROUPS][NUM_REGIONS] = {
        {5.218e-01f, 0.000e+00f, 0.000e+00f},
        {2.264e-01f, 0.000e+00f, 0.000e+00f},
        {4.439e-02f, 0.000e+00f, 0.000e+00f},
        {2.694e-02f, 0.000e+00f, 0.000e+00f},
        {2.107e-02f, 0.000e+00f, 0.000e+00f},
        {6.647e-03f, 0.000e+00f, 0.000e+00f},
        {2.226e-03f, 0.000e+00f, 0.000e+00f},
        {1.093e-03f, 0.000e+00f, 0.000e+00f},
        {5.369e-03f, 0.000e+00f, 0.000e+00f},
        {1.426e-02f, 0.000e+00f, 0.000e+00f}
    };
    static const float sigma_c[NUM_GROUPS][NUM_REGIONS] = {
        {1.678e-01f, 1.770e-02f, 8.331e-02f},
        {7.488e-02f, 8.420e-03f, 3.942e-02f},
        {2.147e-02f, 2.463e-03f, 1.147e-02f},
        {1.200e-01f, 7.635e-04f, 3.548e-03f},
        {7.411e-02f, 2.361e-04f, 1.119e-03f},
        {2.861e-02f, 6.725e-05f, 3.517e-04f},
        {1.511e-02f, 2.104e-04f, 1.070e-04f},
        {4.756e-03f, 1.128e-04f, 3.086e-05f},
        {2.048e-03f, 3.084e-05f, 1.332e-05f},
        {1.990e-03f, 1.366e-03f, 1.324e-03f}
    };
    static const float sigma_s[NUM_GROUPS][NUM_REGIONS] = {
        {4.121e-01f, 8.862e-02f, 4.157e+00f},
        {4.039e-01f, 8.649e-02f, 2.577e+00f},
        {3.967e-01f, 8.593e-02f, 1.608e+00f},
        {4.037e-01f, 8.583e-02f, 1.498e+00f},
        {5.282e-01f, 8.540e-02f, 1.493e+00f},
        {5.011e-01f, 8.195e-02f, 1.483e+00f},
        {5.005e-01f, 6.418e-02f, 1.396e+00f},
        {4.335e-01f, 3.173e-01f, 9.342e-01f},
        {3.193e-01f, 2.063e-01f, 3.991e-01f},
        {2.550e-01f, 1.371e-01f, 1.851e-01f}
    };

    int group = host_energy_group(energy);
    int material = (region >= 0 && region < NUM_REGIONS) ? region : MODERATOR;
    XS xs;
    xs.sig_f = sigma_f[group][material];
    xs.sig_c = sigma_c[group][material];
    xs.sig_s = sigma_s[group][material];
    xs.sig_t = xs.sig_f + xs.sig_c + xs.sig_s;
    return xs;
}

float cpu_circle_distance(float x, float y, float ux, float uy, float radius) {
    float b = 2.0f * (x * ux + y * uy);
    float c = x * x + y * y - radius * radius;
    float delta = b * b - 4.0f * c;
    if (delta < 0.0f) {
        return FLT_MAX;
    }

    float sqrt_delta = std::sqrt(delta);
    const float eps = 1.0e-9f;
    float first = (-b - sqrt_delta) * 0.5f;
    if (first > eps) {
        return first;
    }
    float second = (-b + sqrt_delta) * 0.5f;
    return second > eps ? second : FLT_MAX;
}

float cpu_flat_distance(float position, float direction, float wall) {
    if (std::fabs(direction) < 1.0e-15f) {
        return FLT_MAX;
    }
    float distance = (wall - position) / direction;
    return distance > 1.0e-9f ? distance : FLT_MAX;
}

void cpu_transport_one(CpuParticle particle,
                       CpuRng &rng,
                       std::vector<CpuFissionSite> &fission_sites,
                       Tallies &tallies) {
    bool need_dir = particle.regionchange == 0;

    while (true) {
        if (need_dir) {
            rng.isotropic_dir(particle.ux, particle.uy);
            need_dir = false;
        }

        XS xs = host_cross_sections(particle.Energy, particle.region);
        if (xs.sig_t <= 0.0f) {
            ++tallies.lost_no_surface;
            return;
        }

        float distance_to_collision = -std::log(rng.uniform()) / xs.sig_t;
        float distance_to_boundary = FLT_MAX;
        int next_region = particle.region;
        bool square_boundary = false;
        int square_axis = 0;

        if (particle.region == FUEL) {
            float distance = cpu_circle_distance(
                particle.x, particle.y, particle.ux, particle.uy, FUEL_RADIUS);
            if (distance < distance_to_boundary) {
                distance_to_boundary = distance;
                next_region = CLAD;
            }
        } else if (particle.region == CLAD) {
            float fuel_distance = cpu_circle_distance(
                particle.x, particle.y, particle.ux, particle.uy, FUEL_RADIUS);
            float clad_distance = cpu_circle_distance(
                particle.x, particle.y, particle.ux, particle.uy, DEFAULT_GEOMETRY.r_clad_out);
            if (fuel_distance <= clad_distance) {
                distance_to_boundary = fuel_distance;
                next_region = FUEL;
            } else {
                distance_to_boundary = clad_distance;
                next_region = MODERATOR;
            }
        } else {
            float clad_distance = cpu_circle_distance(
                particle.x, particle.y, particle.ux, particle.uy, DEFAULT_GEOMETRY.r_clad_out);
            if (clad_distance < distance_to_boundary) {
                distance_to_boundary = clad_distance;
                next_region = CLAD;
            }

            float half_pitch = 0.5f * DEFAULT_GEOMETRY.pitch;
            auto try_wall = [&](float position, float direction, float wall, int axis) {
                float distance = cpu_flat_distance(position, direction, wall);
                if (distance < distance_to_boundary) {
                    distance_to_boundary = distance;
                    square_boundary = true;
                    square_axis = axis;
                }
            };
            if (particle.ux > 0.0f) try_wall(particle.x, particle.ux, half_pitch, 0);
            if (particle.ux < 0.0f) try_wall(particle.x, particle.ux, -half_pitch, 0);
            if (particle.uy > 0.0f) try_wall(particle.y, particle.uy, half_pitch, 1);
            if (particle.uy < 0.0f) try_wall(particle.y, particle.uy, -half_pitch, 1);
        }

        if (distance_to_boundary == FLT_MAX) {
            ++tallies.lost_no_surface;
            return;
        }

        if (distance_to_collision >= distance_to_boundary) {
            particle.x += distance_to_boundary * particle.ux;
            particle.y += distance_to_boundary * particle.uy;

            if (square_boundary) {
                if (square_axis == 0) {
                    particle.ux = -particle.ux;
                } else {
                    particle.uy = -particle.uy;
                }
                particle.region = MODERATOR;
            } else {
                const float boundary_eps = 1.0e-4f;
                particle.x += boundary_eps * particle.ux;
                particle.y += boundary_eps * particle.uy;
                particle.region = next_region;
            }
        } else {
            particle.x += distance_to_collision * particle.ux;
            particle.y += distance_to_collision * particle.uy;

            float reaction_sample = rng.uniform();
            float fission_probability = xs.sig_f / xs.sig_t;
            float capture_probability = xs.sig_c / xs.sig_t;

            if (reaction_sample <= fission_probability) {
                ++tallies.fission;
                int fission_neutrons = rng.fission_nu();
                tallies.neutrons_produced += static_cast<unsigned long long>(fission_neutrons);
                for (int child = 0; child < fission_neutrons; ++child) {
                    fission_sites.push_back({particle.x, particle.y});
                }
                return;
            }

            if (reaction_sample <= fission_probability + capture_probability) {
                ++tallies.capture;
                return;
            }

            ++tallies.scattering;
            float mass_number = 4.5f;
            if (particle.region == FUEL) {
                mass_number = 238.02891f;
            } else if (particle.region == CLAD) {
                mass_number = 26.981539f;
            }
            float alpha = std::pow((mass_number - 1.0f) / (mass_number + 1.0f), 2.0f);
            particle.Energy *= alpha + (1.0f - alpha) * rng.uniform();
            need_dir = true;
        }
    }
}

void copy_cpu_sites_to_fission_bank(const std::vector<CpuFissionSite> &sites,
                                    NeutronSoA d_fission_bank,
                                    int fission_bank_capacity,
                                    int *d_fission_bank_count,
                                    int &fission_bank_count,
                                    Tallies &global_tallies) {
    if (sites.empty()) {
        return;
    }

    int available = fission_bank_capacity - fission_bank_count;
    int copied = static_cast<int>(sites.size()) < available
        ? static_cast<int>(sites.size())
        : available;

    if (copied > 0) {
        std::vector<float> x(copied);
        std::vector<float> y(copied);
        for (int i = 0; i < copied; ++i) {
            x[i] = sites[i].x;
            y[i] = sites[i].y;
        }
        CUDA_CHECK(cudaMemcpy(d_fission_bank.x + fission_bank_count, x.data(),
                              copied * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fission_bank.y + fission_bank_count, y.data(),
                              copied * sizeof(float), cudaMemcpyHostToDevice));
        fission_bank_count += copied;
        CUDA_CHECK(cudaMemcpy(d_fission_bank_count, &fission_bank_count,
                              sizeof(int), cudaMemcpyHostToDevice));
    }

    if (copied < static_cast<int>(sites.size())) {
        global_tallies.fission_bank_overflow +=
            static_cast<unsigned long long>(sites.size() - copied);
    }
}

} // namespace

int main(int argc, char **argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : DEFAULT_NEUTRONS;
    int num_generations = argc > 2 ? std::atoi(argv[2]) : DEFAULT_GENERATIONS;

    if (N <= 0 || num_generations <= 0 || argc > 3) {
        printf("Usage: %s [neutrons] [generations]\n", argv[0]);
        return 1;
    }

    float r_fuel = FUEL_RADIUS;
    int queue_capacity = N * QUEUE_MULTIPLIER;

    printf("Config: neutrons=%d generations=%d queue_capacity=%d cpu_switch_threshold=%d\n",
           N, num_generations, queue_capacity, CPU_SWITCH_THRESHOLD);

    curandState *d_rng_states;
    CUDA_CHECK(cudaMalloc(&d_rng_states, queue_capacity * sizeof(curandState)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int capacity_blocks = (queue_capacity + threads - 1) / threads;

    init_cross_sections();
    init_rng<<<capacity_blocks, threads>>>(d_rng_states, 1234UL, queue_capacity);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    NeutronSoA d_move_queue;
    NeutronSoA d_next_move_queue;
    NeutronSoA d_collision_queue;
    NeutronSoA d_fission_bank;
    HistoryTallies *d_history_tallies;
    Tallies *d_global_tallies;

    auto alloc_soa = [](NeutronSoA &soa, int capacity) {
        cudaMalloc(&soa.x,            capacity * sizeof(float));
        cudaMalloc(&soa.y,            capacity * sizeof(float));
        cudaMalloc(&soa.Energy,       capacity * sizeof(float));
        cudaMalloc(&soa.ux,           capacity * sizeof(float));
        cudaMalloc(&soa.uy,           capacity * sizeof(float));
        cudaMalloc(&soa.region,       capacity * sizeof(int));
        cudaMalloc(&soa.regionchange, capacity * sizeof(int));
        cudaMalloc(&soa.rng_state,    capacity * sizeof(curandState));
    };

    alloc_soa(d_move_queue,      queue_capacity);
    alloc_soa(d_next_move_queue, queue_capacity);
    alloc_soa(d_collision_queue, queue_capacity);
    int fission_bank_capacity = queue_capacity;
    alloc_soa(d_fission_bank, fission_bank_capacity);

    CUDA_CHECK(cudaMalloc(&d_history_tallies, queue_capacity * sizeof(HistoryTallies)));
    CUDA_CHECK(cudaMalloc(&d_global_tallies, sizeof(Tallies)));
    CUDA_CHECK(cudaMemset(d_global_tallies, 0, sizeof(Tallies)));

    int *d_move_count, *d_next_move_count, *d_collision_count;
    CUDA_CHECK(cudaMalloc(&d_move_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_move_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_count, sizeof(int)));

    int *d_fission_bank_count;
    int fission_bank_count = 0;
    CUDA_CHECK(cudaMalloc(&d_fission_bank_count, sizeof(int)));

    // Scratch buffers for sorting move_queue by region before each move_kernel launch.
    // d_sort_keys: copy of region values used as sort key (so the original array is untouched).
    // d_sort_idx:  permutation produced by the sort, consumed by gather_kernel.
    int *d_sort_keys;
    int *d_sort_idx;
    CUDA_CHECK(cudaMalloc(&d_sort_keys, queue_capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sort_idx,  queue_capacity * sizeof(int)));


    int move_count = 0;

    initialize_neutrons<<<blocks, threads>>>(d_rng_states, d_move_queue, r_fuel, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    move_count = N;
    CUDA_CHECK(cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_fission_bank_count, &fission_bank_count, sizeof(int), cudaMemcpyHostToDevice));

    int completed_generations = 0;
    double keff_sum = 0.0;
    CpuRng cpu_rng(5678UL);
    const int max_iterations = 100000;
    for (int generation = 0; generation < num_generations; ++generation) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_fission_bank_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

        int iter = 0;
        bool completed_on_cpu = false;
        while (move_count > 0) {
            if (move_count <= CPU_SWITCH_THRESHOLD) {
                CUDA_CHECK(cudaDeviceSynchronize());

                std::vector<CpuParticle> cpu_particles(move_count);
                CUDA_CHECK(cudaMemcpy(&fission_bank_count, d_fission_bank_count,
                                      sizeof(int), cudaMemcpyDeviceToHost));

                std::vector<float> host_x(move_count);
                std::vector<float> host_y(move_count);
                std::vector<float> host_energy(move_count);
                std::vector<float> host_ux(move_count);
                std::vector<float> host_uy(move_count);
                std::vector<int> host_region(move_count);
                std::vector<int> host_regionchange(move_count);
                CUDA_CHECK(cudaMemcpy(host_x.data(), d_move_queue.x,
                                      move_count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(host_y.data(), d_move_queue.y,
                                      move_count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(host_energy.data(), d_move_queue.Energy,
                                      move_count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(host_ux.data(), d_move_queue.ux,
                                      move_count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(host_uy.data(), d_move_queue.uy,
                                      move_count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(host_region.data(), d_move_queue.region,
                                      move_count * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(host_regionchange.data(), d_move_queue.regionchange,
                                      move_count * sizeof(int), cudaMemcpyDeviceToHost));

                std::vector<CpuFissionSite> cpu_fission_sites;
                cpu_fission_sites.reserve(static_cast<size_t>(move_count) * 3);
                Tallies cpu_tallies = {};
                for (int i = 0; i < move_count; ++i) {
                    cpu_particles[i] = {
                        host_x[i], host_y[i], host_energy[i], host_ux[i], host_uy[i],
                        host_region[i], host_regionchange[i]
                    };
                    cpu_transport_one(cpu_particles[i], cpu_rng, cpu_fission_sites, cpu_tallies);
                }

                Tallies global_tallies = {};
                CUDA_CHECK(cudaMemcpy(&global_tallies, d_global_tallies,
                                      sizeof(Tallies), cudaMemcpyDeviceToHost));
                global_tallies.fission += cpu_tallies.fission;
                global_tallies.capture += cpu_tallies.capture;
                global_tallies.scattering += cpu_tallies.scattering;
                global_tallies.leakage += cpu_tallies.leakage;
                global_tallies.neutrons_produced += cpu_tallies.neutrons_produced;
                global_tallies.lost_no_surface += cpu_tallies.lost_no_surface;

                copy_cpu_sites_to_fission_bank(
                    cpu_fission_sites,
                    d_fission_bank,
                    fission_bank_capacity,
                    d_fission_bank_count,
                    fission_bank_count,
                    global_tallies
                );
                CUDA_CHECK(cudaMemcpy(d_global_tallies, &global_tallies,
                                      sizeof(Tallies), cudaMemcpyHostToDevice));

                printf("Generation %d CPU tail histories.........=  %d\n", generation, move_count);
                move_count = 0;
                CUDA_CHECK(cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice));
                completed_on_cpu = true;
                break;
            }

            int active_blocks = (move_count + threads - 1) / threads;
#if DEBUG_TRANSPORT
            if (iter % 100 == 0) {
                printf("generation = %d, iter = %d, move_count = %d\n", generation, iter, move_count);
            }
#endif
            if (iter >= max_iterations) {
                printf("stopping: max iterations reached with move_count = %d\n", move_count);
                break;
            }

            reset_counts_kernel<<<1, 1>>>(d_next_move_count, d_collision_count);
            CUDA_CHECK(cudaGetLastError());

            if (iter % SORT_INTERVAL == 0) {
                thrust::device_ptr<int> keys(d_sort_keys);
                thrust::device_ptr<int> idx(d_sort_idx);
                thrust::copy(thrust::device,
                    thrust::device_ptr<int>(d_move_queue.region),
                    thrust::device_ptr<int>(d_move_queue.region) + move_count,
                    keys);
                thrust::sequence(thrust::device, idx, idx + move_count);
                thrust::sort_by_key(thrust::device, keys, keys + move_count, idx);
                gather_kernel<<<active_blocks, threads>>>(
                    d_move_queue, d_sort_idx, move_count, d_next_move_queue);
                NeutronSoA tmp    = d_move_queue;
                d_move_queue      = d_next_move_queue;
                d_next_move_queue = tmp;
                reset_counts_kernel<<<1, 1>>>(d_next_move_count, d_collision_count);
                CUDA_CHECK(cudaGetLastError());
            }

            move_kernel<<<active_blocks, threads>>>(
                d_move_queue, d_move_count,
                d_next_move_queue, d_next_move_count,
                d_collision_queue, d_collision_count,
                d_global_tallies,
                queue_capacity,
                r_fuel
            );
            CUDA_CHECK(cudaGetLastError());

            collision_kernel<<<active_blocks, threads>>>(
                d_collision_queue, d_collision_count,
                d_next_move_queue, d_next_move_count,
                d_fission_bank, fission_bank_capacity,
                d_fission_bank_count,
                d_history_tallies,
                d_global_tallies,
                queue_capacity
            );
            CUDA_CHECK(cudaGetLastError());

            NeutronSoA temp_queue = d_move_queue;
            d_move_queue      = d_next_move_queue;
            d_next_move_queue = temp_queue;

            int *temp_count = d_move_count;
            d_move_count = d_next_move_count;
            d_next_move_count = temp_count;

            ++iter;

            CUDA_CHECK(cudaMemcpy(&move_count, d_move_count, sizeof(int), cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaMemcpy(&fission_bank_count, d_fission_bank_count, sizeof(int), cudaMemcpyDeviceToHost));
        double generation_keff = static_cast<double>(fission_bank_count) / static_cast<double>(N);
        printf("Generation %d Fission Bank Sites.........=  %d\n", generation, fission_bank_count);
        if (completed_on_cpu) {
            printf("Generation %d Completed With CPU Tail.....=  yes\n", generation);
        }
        printf("Generation %d keff estimate..............=  %.12f\n", generation, generation_keff);
        keff_sum += generation_keff;
        completed_generations = generation + 1;

        if (generation + 1 >= num_generations || fission_bank_count <= 0) {
            break;
        }

        resample_kernel<<<blocks, threads>>>(
            d_fission_bank,
            fission_bank_count,
            d_move_queue,
            N,
            d_rng_states
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        move_count = N;
        CUDA_CHECK(cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice));
        reset_counts_kernel<<<1, 1>>>(d_next_move_count, d_collision_count);
        CUDA_CHECK(cudaGetLastError());
    }

    Tallies global_tallies = {};
    CUDA_CHECK(cudaMemcpy(&global_tallies, d_global_tallies, sizeof(Tallies), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&fission_bank_count, d_fission_bank_count, sizeof(int), cudaMemcpyDeviceToHost));

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
        printf("Average keff.............................=  %.12f\n", keff_sum / static_cast<double>(completed_generations));
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
    printf("Queue Overflowed Particles...............=  %llu\n", global_tallies.queue_overflow);
    printf("Fission Bank Overflowed Particles........=  %llu\n", global_tallies.fission_bank_overflow);
    printf("Lost With No Surface.....................=  %llu\n", global_tallies.lost_no_surface);
#if DEBUG_TRANSPORT
    printf("  Lost No Surface Fuel...................=  %llu\n", global_tallies.lost_no_surface_fuel);
    printf("  Lost No Surface Clad...................=  %llu\n", global_tallies.lost_no_surface_clad);
    printf("  Lost No Surface Moderator..............=  %llu\n", global_tallies.lost_no_surface_moderator);
    printf("  Lost No Surface Valid Region...........=  %llu\n", global_tallies.lost_no_surface_valid_region);
    printf("  Lost No Surface Invalid Region.........=  %llu\n", global_tallies.lost_no_surface_invalid_region);
    printf("Fuel Surface Crossings...................=  %llu\n", global_tallies.fuel_surface_crossings);
    printf("Clad Surface Crossings...................=  %llu\n", global_tallies.clad_surface_crossings);
    printf("Square Surface Crossings.................=  %llu\n", global_tallies.square_surface_crossings);
#endif

    auto free_soa = [](NeutronSoA &soa) {
        cudaFree(soa.x);
        cudaFree(soa.y);
        cudaFree(soa.Energy);
        cudaFree(soa.ux);
        cudaFree(soa.uy);
        cudaFree(soa.region);
        cudaFree(soa.regionchange);
        cudaFree(soa.rng_state);
    };

    CUDA_CHECK(cudaFree(d_sort_idx));
    CUDA_CHECK(cudaFree(d_sort_keys));
    CUDA_CHECK(cudaFree(d_fission_bank_count));
    CUDA_CHECK(cudaFree(d_global_tallies));
    free_soa(d_fission_bank);
    CUDA_CHECK(cudaFree(d_history_tallies));
    free_soa(d_collision_queue);
    free_soa(d_next_move_queue);
    free_soa(d_move_queue);
    CUDA_CHECK(cudaFree(d_rng_states));

    printf("simulation ended!\n");
    return 0;
}
