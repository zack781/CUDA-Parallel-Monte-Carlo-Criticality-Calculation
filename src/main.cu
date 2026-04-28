#include "include/sim.cuh"
#include "common.h"
#include "transport.cu"
#include "rng.cu"

#define Neutrons_Number 100
#define FUEL_RADIUS 0.53

// move_queue: contains neutrons that need to be moved to their next event
// next_move_queue: contains neutrons that hit a geometric boundary and need transport in the next iteration
// collision_queue: contains neutrons that reached their collision site

int main() {
    int N = Neutrons_Number;
    float r_fuel = FUEL_RADIUS;

    curandState *d_rng_states;
    Neutron *d_source_particles;
    cudaMalloc(&d_source_particles, N * sizeof(Neutron));

    cudaMalloc(&d_rng_states, N * sizeof(curandState));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    init_rng<<<blocks, threads>>>(d_rng_states, 1234UL, N);

    // Initialize
    Neutron *d_move_queue;
    Neutron *d_next_move_queue;
    Neutron *d_collision_queue;
    Neutron *d_fission_bank;
    HistoryTallies *d_history_tallies;

    cudaMalloc(&d_move_queue, N * sizeof(Neutron));
    cudaMalloc(&d_next_move_queue, N * sizeof(Neutron));
    cudaMalloc(&d_collision_queue, N * sizeof(Neutron));
    int fission_bank_capacity = N * 10;
    cudaMalloc(&d_fission_bank, fission_bank_capacity * sizeof(Neutron));
    cudaMalloc(&d_history_tallies, N * sizeof(HistoryTallies));

    int *d_move_count, *d_next_move_count, *d_collision_count;
    cudaMalloc(&d_move_count, sizeof(int));
    cudaMalloc(&d_next_move_count, sizeof(int));
    cudaMalloc(&d_collision_count, sizeof(int));

    int *d_fission_bank_count;
    int fission_bank_count = 0;
    cudaMalloc(&d_fission_bank_count, sizeof(int));


    int move_count = 0;
    int next_move_count = 0;
    int collision_count = 0;

    initialize_neutrons<<<blocks, threads>>>(d_rng_states, d_source_particles, r_fuel, N);
    cudaDeviceSynchronize();
    // cudaMemcpy(d_move_queue, neutrons, N * sizeof(Neutron), H2D);
    cudaMemcpy(d_move_queue, d_source_particles, N * sizeof(Neutron), cudaMemcpyDeviceToDevice);
    move_count = N;
    cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_fission_bank_count, &fission_bank_count, sizeof(int), cudaMemcpyHostToDevice);

    while (move_count > 0) {
        printf("move_count = %d\n", move_count);
        if (move_count > 0) {
            int zero = 0;
            cudaMemcpy(d_next_move_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            move_kernel<<<blocks, threads>>>(
                d_move_queue, d_move_count,
                d_next_move_queue, d_next_move_count,
                d_collision_queue, d_collision_count,
                d_rng_states, r_fuel
            );
            cudaDeviceSynchronize();

            cudaMemcpy(&next_move_count, d_next_move_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&collision_count, d_collision_count, sizeof(int), cudaMemcpyDeviceToHost);
        }

        if (collision_count > 0) {
            collision_kernel<<<blocks, threads>>>(
                d_collision_queue, collision_count,
                d_next_move_queue, d_next_move_count,
                d_fission_bank, fission_bank_capacity,
                d_fission_bank_count,
                d_history_tallies,
                d_rng_states
            );
            cudaDeviceSynchronize();
            cudaMemcpy(&next_move_count, d_next_move_count, sizeof(int), cudaMemcpyDeviceToHost);
        }

        // Sorting and consolidation here
        // // Phase 3: CONSOLIDATE (MISSING!)

        // Merge scattered + boundary + fission into move_queue for next iteration
        // if (fission_bank_count > 0 || move_count > 0) {
            // Copy scattered particles from move_queue
            // Copy boundary particles from next_move_queue
            // Copy fission from fission_bank
            // Into a single consolidated move_queue
            // Update move_count = scattered + boundary + fission
        // }
        int zero = 0;

        Neutron *temp = d_move_queue;
        d_move_queue = d_next_move_queue;
        d_next_move_queue = temp;

        move_count = next_move_count;
        next_move_count = 0;
        collision_count = 0;

        cudaMemcpy(d_move_count, &move_count, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_next_move_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaFree(d_fission_bank_count);
    cudaFree(d_fission_bank);
    cudaFree(d_history_tallies);
    cudaFree(d_collision_queue);
    cudaFree(d_next_move_queue);
    cudaFree(d_move_queue);
    cudaFree(d_rng_states);
    cudaFree(d_source_particles);

    printf("simulation ended!\n");
    return 0;
}
