__global__ void transport_kernel(int neutrons_number, float * Neutrons_Energy, float * theta1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Thread ID = neutron ID
    if (i >= n_neutrons) return;


    int alive        = 1;
    int interaction  = 0;
    int boundary     = 0;
    int regionchange = 0;
    int surface      = 0;
    int region       = 0;

    // malloc for an E array
    // append Neutrons_Energy[i] to E


    // malloc for x array
    // malloc for y array
    // x <- r[i] * cos(theta1[i])
    // y <- r[i] * sin(theta1[i])
    while (alive == 1) {
        // Transport neutron i (same logic as Python)
        // Check boundaries, sample interactions
        // Update tallies (with atomics)
    }
}
