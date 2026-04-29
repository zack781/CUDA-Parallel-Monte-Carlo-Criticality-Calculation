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
    int queue_capacity = N * 10;

    curandState *d_rng_states;
    Neutron *d_source_particles;
    cudaMalloc(&d_source_particles, N * sizeof(Neutron));

    cudaMalloc(&d_rng_states, queue_capacity * sizeof(curandState));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int capacity_blocks = (queue_capacity + threads - 1) / threads;

    init_rng<<<capacity_blocks, threads>>>(d_rng_states, 1234UL, queue_capacity);

    // Initialize
    Neutron *d_move_queue;
    Neutron *d_next_move_queue;
    Neutron *d_collision_queue;
    Neutron *d_fission_bank;
    HistoryTallies *d_history_tallies;
    Tallies *d_global_tallies;

    cudaMalloc(&d_move_queue, queue_capacity * sizeof(Neutron));
    cudaMalloc(&d_next_move_queue, queue_capacity * sizeof(Neutron));
    cudaMalloc(&d_collision_queue, queue_capacity * sizeof(Neutron));
    int fission_bank_capacity = queue_capacity;
    cudaMalloc(&d_fission_bank, fission_bank_capacity * sizeof(Neutron));
    cudaMalloc(&d_history_tallies, queue_capacity * sizeof(HistoryTallies));
    cudaMalloc(&d_global_tallies, sizeof(Tallies));
    cudaMemset(d_global_tallies, 0, sizeof(Tallies));

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
            int move_blocks = (move_count + threads - 1) / threads;
            int zero = 0;
            cudaMemcpy(d_next_move_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_collision_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
            move_kernel<<<move_blocks, threads>>>(
                d_move_queue, d_move_count,
                d_next_move_queue, d_next_move_count,
                d_collision_queue, d_collision_count,
                d_global_tallies,
                r_fuel
            );
            cudaDeviceSynchronize();

            cudaMemcpy(&next_move_count, d_next_move_count, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&collision_count, d_collision_count, sizeof(int), cudaMemcpyDeviceToHost);
        }

        if (collision_count > 0) {
            int collision_blocks = (collision_count + threads - 1) / threads;
            collision_kernel<<<collision_blocks, threads>>>(
                d_collision_queue, collision_count,
                d_next_move_queue, d_next_move_count,
                d_fission_bank, fission_bank_capacity,
                d_fission_bank_count,
                d_history_tallies,
                d_global_tallies
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

    Tallies global_tallies = {};
    cudaMemcpy(&global_tallies, d_global_tallies, sizeof(Tallies), cudaMemcpyDeviceToHost);

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
    printf("Number of Interactions...................=  %llu\n", interactions);
    printf("Number of Scattering Events..............=  %llu\n", global_tallies.scattering);
    printf("Number of Capture Events.................=  %llu\n", global_tallies.capture);
    printf("Number of Fission Events.................=  %llu\n", global_tallies.fission);
    printf("Number of Absorption Events..............=  %llu\n", absorption);
    printf("Average nu...............................=  %.6f\n", average_nu);
    printf("Number of Neutrons Produced by Fission...=  %llu\n", global_tallies.neutrons_produced);
    printf("Number of Neutrons Leaked from System....=  %llu\n", global_tallies.leakage);
    printf("Effective Multiplication Factor(keff)....=  %.12f\n", keff);

    cudaFree(d_fission_bank_count);
    cudaFree(d_global_tallies);
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
