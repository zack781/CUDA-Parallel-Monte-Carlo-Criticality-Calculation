#define DEBUG_TRANSPORT 0

#include "include/sim.cuh"
#include "common.h"
#include "transport.cu"
#include "rng.cu"
#include "fission_bank.cu"

#define Neutrons_Number 1000
#define NUM_GENERATIONS 10
#define FUEL_RADIUS 0.53

// move_queue: contains neutrons that need to be moved to their next event
// next_move_queue: contains neutrons that hit a geometric boundary and need transport in the next iteration
// collision_queue: contains neutrons that reached their collision site

int main() {
    int N = Neutrons_Number;
    float r_fuel = FUEL_RADIUS;
    int queue_capacity = N * 10;

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

    int *d_move_count, *d_next_move_count, *d_collision_count;
    CUDA_CHECK(cudaMalloc(&d_move_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_move_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_collision_count, sizeof(int)));

    int *d_fission_bank_count;
    int fission_bank_count = 0;
    CUDA_CHECK(cudaMalloc(&d_fission_bank_count, sizeof(int)));


    int move_count = 0;
    int next_move_count = 0;
    int collision_count = 0;

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
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_fission_bank_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

        int iter = 0;
        while (move_count > 0) {
#if DEBUG_TRANSPORT
            if (iter % 100 == 0) {
                printf("generation = %d, iter = %d, move_count = %d\n", generation, iter, move_count);
            }
#endif
            if (iter >= max_iterations) {
                printf("stopping: max iterations reached with move_count = %d\n", move_count);
                break;
            }
            iter++;

            if (move_count > 0) {
                int move_blocks = (move_count + threads - 1) / threads;
                CUDA_CHECK(cudaMemcpy(d_next_move_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
                move_kernel<<<move_blocks, threads>>>(
                    d_move_queue, d_move_count,
                    d_next_move_queue, d_next_move_count,
                    d_collision_queue, d_collision_count,
                    d_global_tallies,
                    queue_capacity,
                    r_fuel
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(&next_move_count, d_next_move_count, sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&collision_count, d_collision_count, sizeof(int), cudaMemcpyDeviceToHost));
            }

            if (collision_count > 0) {
                int collision_blocks = (collision_count + threads - 1) / threads;
                collision_kernel<<<collision_blocks, threads>>>(
                    d_collision_queue, collision_count,
                    d_next_move_queue, d_next_move_count,
                    d_fission_bank, fission_bank_capacity,
                    d_fission_bank_count,
                    d_history_tallies,
                    d_global_tallies,
                    queue_capacity
                );
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(&next_move_count, d_next_move_count, sizeof(int), cudaMemcpyDeviceToHost));
            }

            Neutron *temp = d_move_queue;
            d_move_queue = d_next_move_queue;
            d_next_move_queue = temp;

            move_count = next_move_count;
            next_move_count = 0;
            collision_count = 0;

            CUDA_CHECK(cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_next_move_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        }

        CUDA_CHECK(cudaMemcpy(&fission_bank_count, d_fission_bank_count, sizeof(int), cudaMemcpyDeviceToHost));
        double generation_keff = static_cast<double>(fission_bank_count) / static_cast<double>(N);
        printf("Generation %d Fission Bank Sites.........=  %d\n", generation, fission_bank_count);
        printf("Generation %d keff estimate..............=  %.12f\n", generation, generation_keff);
        completed_generations = generation + 1;

        if (generation + 1 >= NUM_GENERATIONS || fission_bank_count <= 0) {
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
        next_move_count = 0;
        collision_count = 0;
        CUDA_CHECK(cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next_move_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice));
    }

    Tallies global_tallies = {};
    CUDA_CHECK(cudaMemcpy(&global_tallies, d_global_tallies, sizeof(Tallies), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&fission_bank_count, d_fission_bank_count, sizeof(int), cudaMemcpyDeviceToHost));

    unsigned long long interactions =
        global_tallies.scattering + global_tallies.capture + global_tallies.fission;
    unsigned long long absorption = global_tallies.capture + global_tallies.fission;
    unsigned long long neutrons_lost =
        global_tallies.leakage + absorption + global_tallies.fission;
    double average_nu = global_tallies.fission > 0
        ? static_cast<double>(global_tallies.neutrons_produced) /
              static_cast<double>(global_tallies.fission)
        : 0.0;
    double keff = neutrons_lost > 0
        ? static_cast<double>(global_tallies.neutrons_produced + global_tallies.leakage) /
              static_cast<double>(neutrons_lost)
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
    printf("Effective Multiplication Factor(keff)....=  %.12f\n", keff);

    CUDA_CHECK(cudaFree(d_fission_bank_count));
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
