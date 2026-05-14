#define DEBUG_TRANSPORT 0

#include "include/sim.cuh"
#include "common.h"
#include "transport.cu"
#include "rng.cu"
#include "fission_bank.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>

#define DEFAULT_NEUTRONS 10000
#define DEFAULT_GENERATIONS 10
#define DEFAULT_BATCH_SIZE 10
#define QUEUE_MULTIPLIER 10
#define FUEL_RADIUS 0.53

struct ActiveCountSample {
    int generation;
    int iteration;
    int move_count;
};

// move_queue: contains neutrons that need to be moved to their next event
// next_move_queue: contains neutrons that hit a geometric boundary and need transport in the next iteration
// collision_queue: contains neutrons that reached their collision site

int main(int argc, char **argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : DEFAULT_NEUTRONS;
    int num_generations = argc > 2 ? std::atoi(argv[2]) : DEFAULT_GENERATIONS;
    int batch_size = argc > 3 ? std::atoi(argv[3]) : DEFAULT_BATCH_SIZE;
#if PROFILE_HISTORY_LENGTHS || PROFILE_ACTIVE_COUNTS
    int next_profile_arg = 4;
#endif
#if PROFILE_HISTORY_LENGTHS
    const char *history_counts_path =
        argc > next_profile_arg ? argv[next_profile_arg++] : "history_move_counts.csv";
#endif
#if PROFILE_ACTIVE_COUNTS
    const char *active_counts_path =
        argc > next_profile_arg ? argv[next_profile_arg++] : "active_counts.csv";
#endif

    if (N <= 0 || num_generations <= 0 || batch_size <= 0) {
        printf("Usage: %s [neutrons] [generations] [batch_size]\n", argv[0]);
        return 1;
    }

    float r_fuel = FUEL_RADIUS;
    int queue_capacity = N * QUEUE_MULTIPLIER;

    printf("Config: neutrons=%d generations=%d batch_size=%d queue_capacity=%d\n",
           N, num_generations, batch_size, queue_capacity);

    curandState *d_rng_states;
    Neutron *d_source_particles;
    CUDA_CHECK(cudaMalloc(&d_source_particles, N * sizeof(Neutron)));

    CUDA_CHECK(cudaMalloc(&d_rng_states, queue_capacity * sizeof(curandState)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int capacity_blocks = (queue_capacity + threads - 1) / threads;

    init_rng<<<capacity_blocks, threads>>>(d_rng_states, 1234UL, queue_capacity);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize
    Neutron *d_move_queue;
    Neutron *d_next_move_queue;
    Neutron *d_collision_queue;
    Neutron *d_fission_bank;
    HistoryTallies *d_history_tallies;
    Tallies *d_global_tallies;

    CUDA_CHECK(cudaMalloc(&d_move_queue, queue_capacity * sizeof(Neutron)));
    CUDA_CHECK(cudaMalloc(&d_next_move_queue, queue_capacity * sizeof(Neutron)));
    CUDA_CHECK(cudaMalloc(&d_collision_queue, queue_capacity * sizeof(Neutron)));
    int fission_bank_capacity = queue_capacity;
    CUDA_CHECK(cudaMalloc(&d_fission_bank, fission_bank_capacity * sizeof(Neutron)));
    CUDA_CHECK(cudaMalloc(&d_history_tallies, queue_capacity * sizeof(HistoryTallies)));
    CUDA_CHECK(cudaMalloc(&d_global_tallies, sizeof(Tallies)));
    CUDA_CHECK(cudaMemset(d_global_tallies, 0, sizeof(Tallies)));

#if PROFILE_HISTORY_LENGTHS
    unsigned int *d_history_move_counts = nullptr;
    int total_history_count = N * num_generations;
    CUDA_CHECK(cudaMalloc(&d_history_move_counts, total_history_count * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_history_move_counts, 0, total_history_count * sizeof(unsigned int)));
#endif

    int *d_move_count, *d_next_move_count, *d_collision_count;
    CUDA_CHECK(cudaMalloc(&d_move_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_move_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_count, sizeof(int)));

    int *d_fission_bank_count;
    int fission_bank_count = 0;
    CUDA_CHECK(cudaMalloc(&d_fission_bank_count, sizeof(int)));


    int move_count = 0;

    initialize_neutrons<<<blocks, threads>>>(d_rng_states, d_source_particles, r_fuel, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // cudaMemcpy(d_move_queue, neutrons, N * sizeof(Neutron), H2D);
    CUDA_CHECK(cudaMemcpy(d_move_queue, d_source_particles, N * sizeof(Neutron), cudaMemcpyDeviceToDevice));
    move_count = N;
    CUDA_CHECK(cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_fission_bank_count, &fission_bank_count, sizeof(int), cudaMemcpyHostToDevice));

    int completed_generations = 0;
    const int max_iterations = 100000;
#if PROFILE_ACTIVE_COUNTS
    std::vector<ActiveCountSample> active_count_samples;
    active_count_samples.reserve(num_generations * 512);
#endif
    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));
    CUDA_CHECK(cudaEventRecord(gpu_start));

    for (int generation = 0; generation < num_generations; ++generation) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_fission_bank_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

        int iter = 0;
#if PROFILE_ACTIVE_COUNTS
        active_count_samples.push_back({generation, iter, move_count});
#endif
        while (move_count > 0) {
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

            for (int step = 0; step < batch_size && iter < max_iterations; ++step, ++iter) {
                CUDA_CHECK(cudaMemcpy(d_next_move_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
                move_kernel<<<active_blocks, threads>>>(
                    d_move_queue, d_move_count,
                    d_next_move_queue, d_next_move_count,
                    d_collision_queue, d_collision_count,
                    d_global_tallies,
#if PROFILE_HISTORY_LENGTHS
                    d_history_move_counts,
                    total_history_count,
#endif
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

                Neutron *temp_queue = d_move_queue;
                d_move_queue = d_next_move_queue;
                d_next_move_queue = temp_queue;

                int *temp_count = d_move_count;
                d_move_count = d_next_move_count;
                d_next_move_count = temp_count;
            }

            CUDA_CHECK(cudaMemcpy(&move_count, d_move_count, sizeof(int), cudaMemcpyDeviceToHost));
#if PROFILE_ACTIVE_COUNTS
            active_count_samples.push_back({generation, iter, move_count});
#endif
            CUDA_CHECK(cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMemcpy(&fission_bank_count, d_fission_bank_count, sizeof(int), cudaMemcpyDeviceToHost));
        double generation_keff = static_cast<double>(fission_bank_count) / static_cast<double>(N);
        printf("Generation %d Fission Bank Sites.........=  %d\n", generation, fission_bank_count);
        printf("Generation %d keff estimate..............=  %.12f\n", generation, generation_keff);
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
#if PROFILE_HISTORY_LENGTHS
            ,
            (generation + 1) * N
#endif
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        move_count = N;
        CUDA_CHECK(cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next_move_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaEventRecord(gpu_stop));
    CUDA_CHECK(cudaEventSynchronize(gpu_stop));
    float gpu_elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_elapsed_ms, gpu_start, gpu_stop));

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
    printf("GPU Timed Section Seconds................=  %.6f\n", gpu_elapsed_ms / 1000.0f);
#if PROFILE_ACTIVE_COUNTS
    FILE *active_file = std::fopen(active_counts_path, "w");
    if (active_file == nullptr) {
        std::perror("failed to open active count CSV");
    } else {
        std::fprintf(active_file, "generation,iteration,move_count,active_fraction\n");
        for (const ActiveCountSample &sample : active_count_samples) {
            std::fprintf(active_file, "%d,%d,%d,%.9f\n",
                         sample.generation,
                         sample.iteration,
                         sample.move_count,
                         static_cast<double>(sample.move_count) / static_cast<double>(N));
        }
        std::fclose(active_file);
        printf("Active Count CSV.........................=  %s\n", active_counts_path);
    }
#endif
#if PROFILE_HISTORY_LENGTHS
    int written_history_count = completed_generations * N;
    std::vector<unsigned int> history_move_counts(written_history_count);
    CUDA_CHECK(cudaMemcpy(history_move_counts.data(), d_history_move_counts,
                          written_history_count * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    FILE *history_file = std::fopen(history_counts_path, "w");
    if (history_file == nullptr) {
        std::perror("failed to open history move count CSV");
    } else {
        std::fprintf(history_file, "history_id,generation,source_index,move_kernel_calls\n");
        for (int history_id = 0; history_id < written_history_count; ++history_id) {
            std::fprintf(history_file, "%d,%d,%d,%u\n",
                         history_id,
                         history_id / N,
                         history_id % N,
                         history_move_counts[history_id]);
        }
        std::fclose(history_file);
        printf("History Move Count CSV...................=  %s\n", history_counts_path);
    }
#endif
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

    CUDA_CHECK(cudaFree(d_fission_bank_count));
#if PROFILE_HISTORY_LENGTHS
    CUDA_CHECK(cudaFree(d_history_move_counts));
#endif
    CUDA_CHECK(cudaFree(d_global_tallies));
    CUDA_CHECK(cudaFree(d_fission_bank));
    CUDA_CHECK(cudaFree(d_history_tallies));
    CUDA_CHECK(cudaFree(d_collision_queue));
    CUDA_CHECK(cudaFree(d_next_move_queue));
    CUDA_CHECK(cudaFree(d_move_queue));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_source_particles));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));
    CUDA_CHECK(cudaEventDestroy(gpu_start));

    printf("simulation ended!\n");
    return 0;
}
