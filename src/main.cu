#define DEBUG_TRANSPORT 0

#include "include/sim.cuh"
#include "common.h"
#include "transport.cu"
#include "rng.cu"
#include "fission_bank.cu"
#include <cstdlib>
#include <cstdio>

#define DEFAULT_NEUTRONS 10000
#define DEFAULT_GENERATIONS 10
#define QUEUE_MULTIPLIER 10
#define FUEL_RADIUS 0.53

// move_queue: contains neutrons that need to be moved to their next event
// next_move_queue: contains neutrons that hit a geometric boundary and need transport in the next iteration
// collision_queue: contains neutrons that reached their collision site

int main(int argc, char **argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : DEFAULT_NEUTRONS;
    int num_generations = argc > 2 ? std::atoi(argv[2]) : DEFAULT_GENERATIONS;
    float tail_fraction = argc > 3 ? std::atof(argv[3]) : 0.0f;

    if (N <= 0 || num_generations <= 0 ||
        tail_fraction < 0.0f || tail_fraction >= 1.0f || argc > 4) {
        printf("Usage: %s [neutrons] [generations] [tail_fraction]\n", argv[0]);
        return 1;
    }

    float r_fuel = FUEL_RADIUS;
    int queue_capacity = N * QUEUE_MULTIPLIER;
    int tail_cutoff = static_cast<int>(tail_fraction * N);
    if (tail_fraction > 0.0f && tail_cutoff < 1) {
        tail_cutoff = 1;
    }

    printf("Config: neutrons=%d generations=%d queue_capacity=%d tail_cutoff=%d\n",
           N, num_generations, queue_capacity, tail_cutoff);

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
    RegionCorrectionTallies *d_region_correction;

    CUDA_CHECK(cudaMalloc(&d_move_queue, queue_capacity * sizeof(Neutron)));
    CUDA_CHECK(cudaMalloc(&d_next_move_queue, queue_capacity * sizeof(Neutron)));
    CUDA_CHECK(cudaMalloc(&d_collision_queue, queue_capacity * sizeof(Neutron)));
    int fission_bank_capacity = queue_capacity;
    CUDA_CHECK(cudaMalloc(&d_fission_bank, fission_bank_capacity * sizeof(Neutron)));
    CUDA_CHECK(cudaMalloc(&d_history_tallies, queue_capacity * sizeof(HistoryTallies)));
    CUDA_CHECK(cudaMalloc(&d_global_tallies, sizeof(Tallies)));
    CUDA_CHECK(cudaMemset(d_global_tallies, 0, sizeof(Tallies)));
    CUDA_CHECK(cudaMalloc(&d_region_correction, sizeof(RegionCorrectionTallies)));

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
    double corrected_keff_sum = 0.0;
    const int max_iterations = 100000;
    for (int generation = 0; generation < num_generations; ++generation) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_fission_bank_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_region_correction, 0, sizeof(RegionCorrectionTallies)));

        int iter = 0;
        while (move_count > tail_cutoff) {
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
            move_kernel<<<active_blocks, threads>>>(
                d_move_queue, d_move_count,
                d_next_move_queue, d_next_move_count,
                d_collision_queue, d_collision_count,
                d_global_tallies,
                d_region_correction,
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
                d_region_correction,
                queue_capacity
            );
            CUDA_CHECK(cudaGetLastError());

            Neutron *temp_queue = d_move_queue;
            d_move_queue = d_next_move_queue;
            d_next_move_queue = temp_queue;

            int *temp_count = d_move_count;
            d_move_count = d_next_move_count;
            d_next_move_count = temp_count;

            ++iter;

            CUDA_CHECK(cudaMemcpy(&move_count, d_move_count, sizeof(int), cudaMemcpyDeviceToHost));
        }

        if (move_count > 0) {
            int tail_blocks = (move_count + threads - 1) / threads;
            tail_correction_kernel<<<tail_blocks, threads>>>(
                d_move_queue,
                d_move_count,
                d_region_correction
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        RegionCorrectionTallies region_correction = {};
        CUDA_CHECK(cudaMemcpy(&region_correction, d_region_correction, sizeof(RegionCorrectionTallies), cudaMemcpyDeviceToHost));

        unsigned long long total_completed = 0;
        unsigned long long total_produced = 0;
        for (int region = 0; region < NUM_REGIONS; ++region) {
            total_completed += region_correction.completed[region];
            total_produced += region_correction.produced[region];
        }

        double global_observed_yield = total_completed > 0
            ? static_cast<double>(total_produced) / static_cast<double>(total_completed)
            : 0.0;
        double expected_tail_fissions = 0.0;
        double region_yield[NUM_REGIONS] = {};
        for (int region = 0; region < NUM_REGIONS; ++region) {
            region_yield[region] = region_correction.completed[region] > 0
                ? static_cast<double>(region_correction.produced[region]) /
                      static_cast<double>(region_correction.completed[region])
                : global_observed_yield;
            expected_tail_fissions +=
                static_cast<double>(region_correction.tail[region]) * region_yield[region];
        }

        CUDA_CHECK(cudaMemcpy(&fission_bank_count, d_fission_bank_count, sizeof(int), cudaMemcpyDeviceToHost));
        double generation_keff = static_cast<double>(fission_bank_count) / static_cast<double>(N);
        double corrected_fission_count = static_cast<double>(fission_bank_count) +
            expected_tail_fissions;
        double corrected_generation_keff = corrected_fission_count / static_cast<double>(N);
        printf("Generation %d Fission Bank Sites.........=  %d\n", generation, fission_bank_count);
        if (move_count > 0) {
            printf("Generation %d Truncated Tail Particles...=  %d\n", generation, move_count);
            printf("Generation %d Tail by Region F/C/M.......=  %u / %u / %u\n", generation,
                   region_correction.tail[FUEL], region_correction.tail[CLAD], region_correction.tail[MODERATOR]);
            printf("Generation %d Region Yields F/C/M........=  %.6f / %.6f / %.6f\n", generation,
                   region_yield[FUEL], region_yield[CLAD], region_yield[MODERATOR]);
            printf("Generation %d Expected Tail Fissions.....=  %.6f\n", generation, expected_tail_fissions);
            printf("Generation %d Corrected keff estimate....=  %.12f\n", generation, corrected_generation_keff);
        }
        printf("Generation %d keff estimate..............=  %.12f\n", generation, generation_keff);
        corrected_keff_sum += corrected_generation_keff;
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
        printf("Average Corrected keff...................=  %.12f\n", corrected_keff_sum / static_cast<double>(completed_generations));
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

    CUDA_CHECK(cudaFree(d_fission_bank_count));
    CUDA_CHECK(cudaFree(d_region_correction));
    CUDA_CHECK(cudaFree(d_global_tallies));
    CUDA_CHECK(cudaFree(d_fission_bank));
    CUDA_CHECK(cudaFree(d_history_tallies));
    CUDA_CHECK(cudaFree(d_collision_queue));
    CUDA_CHECK(cudaFree(d_next_move_queue));
    CUDA_CHECK(cudaFree(d_move_queue));
    CUDA_CHECK(cudaFree(d_rng_states));
    CUDA_CHECK(cudaFree(d_source_particles));

    printf("simulation ended!\n");
    return 0;
}
