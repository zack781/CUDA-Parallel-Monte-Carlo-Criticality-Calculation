#include "include/sim.cuh"
#include "common.h"
#include "transport.cu"
#include "collision.cu"
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
    Neutron *neutrons;

    cudaMalloc(&d_rng_states, N * sizeof(curandState));
    cudaMalloc(&neutrons, N * sizeof(Neutron));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    init_rng<<<blocks, threads>>>(d_rng_states, 1234UL);

    // Initialize
    Neutron *d_move_queue;
    Neutron *d_next_move_queue;
    Neutron *d_collision_queue;
    Neutron *d_fission_bank;
    HistoryTallies *d_history_tallies;

    cudaMalloc(&d_move_queue, N * sizeof(Neutron));
    cudaMalloc(&d_next_move_queue, N * sizeof(Neutron));
    cudaMalloc(&d_collision_queue, N * sizeof(Neutron));
    cudaMalloc(&d_fission_bank, N * sizeof(Neutron));
    cudaMalloc(&d_history_tallies, N * sizeof(HistoryTallies));

    int *d_move_count, *d_next_move_count, *d_collision_count;
    cudaMalloc(&d_move_count, sizeof(int));
    cudaMalloc(&d_next_move_count, sizeof(int));
    cudaMalloc(&d_collision_count, sizeof(int));

    cudaMemcpy(d_move_count, &move_count, sizeof(int), H2D);

    int *d_fission_bank_capacity, *d_fission_bank_count;
    cudaMalloc(&d_fission_bank_capacity, sizeof(int));
    cudaMalloc(&d_fission_bank_count, sizeof(int));


    int move_count = 0;
    int next_move_count = 0;
    int collision_count = 0;

    initialize_neutrons<<<blocks, threads>>>(d_rng_states, neutrons, r_fuel, N);
    cudaMemcpy(move_queue, neutrons, N * sizeof(Neutron), H2D);
    move_count = N;
    cudaMemcpy(d_move_count, &move_count, sizeof(int), H2D);

    cudaMemcpy(d_fission_bank_capacity, &move_count, sizeof(int), H2D);


    cudaDeviceSynchronize();


    while (next_move > 0 || next_move_count > 0 || collision_count > 0) {
        if (move_count > 0) {
            move_kernel<<<blocks, threads>>>(
                d_move_queue, d_move_count,
                d_next_move_queue, &d_next_move_count,
                d_collision_queue, &d_collision_count,
                d_rng_states
            );
        }

        cudaMemcpy(&next_move_count, &d_next_move_count, sizeof(int), D2H);
        cudaMemcpy(&collision_count, &d_collision_count, sizeof(int), D2H);

        if (collision_count > 0) {
            collision_kernel<<<blocks, threads>>>(d_collision_queue, d_collision_count, d_move_queue, d_move_count, d_fission_bank, d_fission_bank_capacity, d_fission_bank_count, d_history_tallies);
            cudaDeviceSynchronize();
            cudaMemcpy(&move_count, &d_move_count, sizeof(int), D2H);
        }

        // Sorting and consolidation here
    }

    collision_count = 0;
    cudaMemset(&d_collision_count, 0, sizeof(int));

    move_queue = next_move_queue;
    move_count = next_move_count;
    next_move_count = 0;
    cudaMemset(&d_next_move_count, 0, sizeof(int));
}
