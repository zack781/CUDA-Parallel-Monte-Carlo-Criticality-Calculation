#include <stdio.h>
#include <cuda.h>

// #include "common.h"

__global__ void neutron_transport() {
    // Neutron transport kernel: sample collision type, free-flight distance, boundary check
    printf("Neutron Transport Kernel\n");
}

__global__ void fission_bank() {
    // Fission bank kernel: store sites, weight accumulation, k_eff estimator
    printf("Fission Bank Kernel\n");
}

int main() {
    // <<<no. of blocks, no. of threads in block>>>
    neutron_transport<<<1, 1>>>();
    fission_bank<<<1, 1>>>();
    return 0;
}
