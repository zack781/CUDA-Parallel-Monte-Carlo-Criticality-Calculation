#include "include/sim.cuh"
#include "common.h"
#include "transport.cu"
#include "collision.cu"
#include "rng.cu"

#define Neutrons_Number 100

int main() {
    int N = Neutrons_Number;
    curandState *d_rng_states;

    cudaMalloc(&d_rng_states, N * sizeof(curandState));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    init_rng<<<blocks, threads>>>(d_rng_states, 1234UL);
    initialize_neutrons<<<blocks, threads>>>(d_rng_states, x, y, energy, r_fuel, N);
    cudaDeviceSynchronize();

    while (particlesAlive(particles) > 0) {
        for (Particle p : particles) {
            if (onBoundary(p)) {
                // launch transport kernel
            }
            if (atSite(p)) {
                // launch collision kernel
            }
        }
        // Sorting and consolidation here
    }
}
