#ifndef SIM_CUH
#define SIM_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef DEBUG_TRANSPORT
#define DEBUG_TRANSPORT 0
#endif

constexpr int NUM_GROUPS = 10;
constexpr int NUM_REGIONS = 3;

struct XS {
    float sig_f;
    float sig_c;
    float sig_s;
    float sig_t;
};

// Energy group lower bounds
__constant__ float d_GROUP_ENERGY[NUM_GROUPS] = {
    3.0e+1f, 3.0e+0f, 3.0e-1f, 3.0e-2f, 3.0e-3f,
    3.0e-4f, 3.0e-5f, 3.0e-6f, 3.0e-7f, 3.0e-8f
};

// fission cross sections in by energy group and region
__constant__ float d_sigma_f[NUM_GROUPS][NUM_REGIONS] = {
    {1.05e-1f, 0.0f, 0.0f},
    {5.96e-2f, 0.0f, 0.0f},
    {6.02e-2f, 0.0f, 0.0f},
    {1.06e-1f, 0.0f, 0.0f},
    {2.46e-1f, 0.0f, 0.0f},
    {2.50e-1f, 0.0f, 0.0f},
    {1.07e-1f, 0.0f, 0.0f},
    {1.28e+0f, 0.0f, 0.0f},
    {9.30e+0f, 0.0f, 0.0f},
    {2.58e+1f, 0.0f, 0.0f}
};

// capture cross sections in by energy group and region
__constant__ float d_sigma_c[NUM_GROUPS][NUM_REGIONS] = {
    {1.41e-6f, 1.71e-2f, 3.34e-6f},
    {1.34e-3f, 7.83e-3f, 3.34e-6f},
    {1.10e-2f, 2.83e-4f, 2.56e-7f},
    {3.29e-2f, 4.52e-6f, 6.63e-7f},
    {8.23e-2f, 1.06e-5f, 2.24e-7f},
    {4.28e-2f, 4.39e-6f, 1.27e-7f},
    {9.90e-2f, 1.25e-5f, 2.02e-7f},
    {2.51e-1f, 3.98e-5f, 6.02e-7f},
    {2.12e+0f, 1.26e-4f, 1.84e-6f},
    {4.30e+0f, 3.95e-4f, 5.76e-6f}
};

// scattering cross sections in by energy group and region
__constant__ float d_sigma_s[NUM_GROUPS][NUM_REGIONS] = {
    {2.76e-1f, 1.44e-1f, 1.27e-2f},
    {3.88e-1f, 1.76e-1f, 7.36e-2f},
    {4.77e-1f, 3.44e-1f, 2.65e-1f},
    {6.88e-1f, 2.66e-1f, 5.72e-1f},
    {9.38e-1f, 2.06e-1f, 6.69e-1f},
    {1.52e+0f, 2.14e-1f, 6.81e-1f},
    {2.30e+0f, 2.23e-1f, 6.82e-1f},
    {2.45e+0f, 2.31e-1f, 6.83e-1f},
    {9.79e+0f, 2.40e-1f, 6.86e-1f},
    {4.36e+1f, 2.41e-1f, 6.91e-1f}
};

__device__ __forceinline__
XS CrossSections(float Energy, int region)
{
    int group = 9;

    #pragma unroll
    for (int g = 0; g < NUM_GROUPS; g++) {
        if (Energy >= d_GROUP_ENERGY[g]) {
            group = g;
            break;
        }
    }

    XS xs;
    xs.sig_f = d_sigma_f[group][region];
    xs.sig_c = d_sigma_c[group][region];
    xs.sig_s = d_sigma_s[group][region];
    xs.sig_t = xs.sig_f + xs.sig_c + xs.sig_s;

    return xs;
}

struct Geometry {
    float r_fuel;     // Fuel pellet radius.
    float r_clad_in;  // Inner cladding radius.
    float r_clad_out; // Outer cladding radius.
    float pitch;      // Square cell width / pin-to-pin spacing.
};

constexpr Geometry DEFAULT_GEOMETRY = {
    0.53f,
    0.53f,
    0.90f,
    1.837f
};

enum Region {
    FUEL = 0,
    CLAD = 1,
    MODERATOR = 2
};

enum EventType {
    EVENT_COLLISION = 0,
    EVENT_BOUNDARY = 1,
    EVENT_DEAD = 2
};

enum SurfaceId {
    SURFACE_FUEL = 0,
    SURFACE_CLAD_INNER = 1,
    SURFACE_CLAD_OUTER = 2,
    SURFACE_X_MIN = 3,
    SURFACE_X_MAX = 4,
    SURFACE_Y_MIN = 5,
    SURFACE_Y_MAX = 6,
    SURFACE_NONE = 7
};

enum ReactionType {
    REACTION_FISSION = 0,
    REACTION_CAPTURE = 1,
    REACTION_SCATTER = 2,
    REACTION_NONE = 3
};

struct Neutron {
    float x;
    float y;
    float Energy;
    float ux;
    float uy;
    int region;
    int regionchange;
    curandState rng_state;
};

// reaction cross sections for one energy group and region
struct CrossSections {
    float fission;
    float capture;
    float scattering;
    float total;
};

struct BoundaryHit {
    float distance;
    SurfaceId surface;
};

struct Event {
    EventType type;
    float distance;
    SurfaceId surface;
    ReactionType reaction;
};

struct Tallies {
    unsigned long long fission;
    unsigned long long capture;
    unsigned long long scattering;
    unsigned long long leakage;
    unsigned long long neutrons_produced;
    unsigned long long queue_overflow;
    unsigned long long fission_bank_overflow;
    unsigned long long lost_no_surface;
#if DEBUG_TRANSPORT
    unsigned long long lost_no_surface_fuel;
    unsigned long long lost_no_surface_clad;
    unsigned long long lost_no_surface_moderator;
    unsigned long long lost_no_surface_invalid_region;
    unsigned long long lost_no_surface_valid_region;
    unsigned long long fuel_surface_crossings;
    unsigned long long clad_surface_crossings;
    unsigned long long square_surface_crossings;
#endif
};

struct SurfaceTallies {
    unsigned int fuel[NUM_GROUPS];
    unsigned int clad[NUM_GROUPS];
};

// Per-history contributions reduced into global tallies later.
struct HistoryTallies {
    unsigned int fission;
    unsigned int capture;
    unsigned int scattering;
    unsigned int leakage;
    unsigned int neutrons_produced;
    unsigned int fuel_surface[NUM_GROUPS];
    unsigned int clad_surface[NUM_GROUPS];
};

__global__ void init_rng(curandState *states, unsigned long seed, int n = -1);

__global__ void initialize_neutrons(
    curandState *states,
    Neutron *neutrons,
    float r_fuel,
    int n
);

__device__ float random_uniform(curandState *state);

__device__ void sample_isotropic_direction(curandState *state, float *ux, float *uy);

__device__ float sample_initial_energy(curandState *state);

__device__ int sample_fission_multiplicity(curandState *state);

__global__ void move_kernel(
    const Neutron *move_queue,
    int *move_count,
    Neutron *next_move_queue,
    int *next_move_count,
    Neutron *collision_queue,
    int *collision_count,
    Tallies *global_tallies,
    int queue_capacity,
    float r_fuel
);

__global__ void collision_kernel(
    const Neutron *collision_queue,
    const int *collision_count,
    Neutron *next_move_queue,
    int *next_move_count,
    Neutron *fission_bank,
    int fission_bank_capacity,
    int *fission_bank_count,
    HistoryTallies *history_tallies,
    Tallies *global_tallies,
    int queue_capacity
);

__global__ void compact_queue_kernel(
    const Neutron *input_queue,
    const int *keep_flags,
    const int *output_offsets,
    int input_count,
    Neutron *output_queue
);

__global__ void normalize_bank_kernel();

__global__ void entropy_tally_kernel();

__global__ void resample_kernel(
    const Neutron *fission_bank,
    int fission_bank_count,
    Neutron *source_particles,
    int source_count,
    curandState *rng_states
);

__global__ void initialize_source_kernel();

float compute_shannon_entropy(int *bins, int count);

#endif
