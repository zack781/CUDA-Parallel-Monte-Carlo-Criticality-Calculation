#include <stdio.h>
#include <cuda.h>

// #include "common.h"

float Neutrons_Produced = 0;
interaction_point_x = []
interaction_point_y = []
float FuelSurfNeuNum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
float CladSurfNeuNum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Group_Energy = [
    3e+1, 3e+0, 3e-1, 3e-2, 3e-3,
    3e-4, 3e-5, 3e-6, 3e-7, 3e-8
]

float Fission    = 0;
float nu         = 0;
float Capture    = 0;
float Absorption = 0;
float Scattering = 0;
float Leakage    = 0;

// =========================================================
//                         Geometry
// =========================================================
float r_fuel     = 0.53;  // Fuel radius (cm)
float r_clad_in  = 0.53;  // Cladding inner radius (cm)
float r_clad_out = 0.90;  // Cladding outer radius (cm)
float pitch      = 1.837;  // Cell pitch (cm)
float t_clad     = r_clad_out - r_clad_in;



__global__ void init_rng_kernel();

__global__ void transport_kernel();

__global__ void normalize_bank_kernel();

__global__ void entropy_tally_kernel();

__global__ void resample_kernel();

__global__ void initialize_source_kernel();

// main.cu
float compute_shannon_entropy(int* bins, int M);


