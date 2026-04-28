#include "include/sim.cuh"
#include "common.h"
#include "transport.cu"
#include "collision.cu"
#include "rng.cu"

#define Neutrons_Number 100
#define FUEL_RADIUS 0.53

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
    initialize_neutrons<<<blocks, threads>>>(d_rng_states, neutrons, r_fuel, N);
    cudaDeviceSynchronize();

    Neutron *move_queue;
    Neutron *next_move_queue;
    Neutron *collision_queue;

    cudaMalloc(&move_queue, N * sizeof(Neutron));
    cudaMalloc(&next_move_queue, N * sizeof(Neutron));
    cudaMalloc(&collision_queue, N * sizeof(Neutron));

    int move_count = 0;
    int next_move_count = 0;
    int collision_count = 0;

    while (next_move_count > 0 || collision_count > 0) {
        for (Neutron p : next_move_queue) {
            // if (onBoundary(p)) {
            //     // launch transport kernel
            // }
            // if (atSite(p)) {
            //     // launch collision kernel
            // }
            move_kernel<<<blocks, threads>>>(move_queue, move_count, next_move_queue, next_move_count, collision_queue, collision_count, d_rng_states);
        }

        for (Neutron p : collision_queue) {
            collision_kernel<<<blocks, threads>>>(collision_queue, collision_count, next_move_queue, next_move_count, fission_bank, fission_bank_capacity, fission_bank_count, history_tallies);
        }

        // Sorting and consolidation here
    }
}
