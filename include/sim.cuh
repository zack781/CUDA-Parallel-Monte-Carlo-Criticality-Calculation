#ifndef SIM_CUH
#define SIM_CUH

#include <cuda_runtime.h>

constexpr int NUM_GROUPS = 10;
constexpr int NUM_REGIONS = 3;
constexpr int NUM_SURFACES = 8;

constexpr float GROUP_ENERGY[NUM_GROUPS] = {
    3.0e+1f, 3.0e+0f, 3.0e-1f, 3.0e-2f, 3.0e-3f,
    3.0e-4f, 3.0e-5f, 3.0e-6f, 3.0e-7f, 3.0e-8f
};

struct Geometry {
    float r_fuel;
    float r_clad_in;
    float r_clad_out;
    float pitch;
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
    float energy;
    float ux;
    float uy;
    int region;
};

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
};

struct SurfaceTallies {
    unsigned int fuel[NUM_GROUPS];
    unsigned int clad[NUM_GROUPS];
};

struct HistoryTallies {
    unsigned int fission;
    unsigned int capture;
    unsigned int scattering;
    unsigned int leakage;
    unsigned int neutrons_produced;
    unsigned int fuel_surface[NUM_GROUPS];
    unsigned int clad_surface[NUM_GROUPS];
};

__device__ float random_uniform(unsigned int *state);

__device__ void sample_isotropic_direction(unsigned int *state, float *ux, float *uy);

__global__ void init_rng_kernel();

__global__ void move_kernel(
    const Neutron *move_queue,
    int move_count,
    Neutron *next_move_queue,
    int *next_move_count,
    Neutron *collision_queue,
    int *collision_count,
    HistoryTallies *history_tallies
);

__global__ void collision_kernel(
    const Neutron *collision_queue,
    int collision_count,
    Neutron *move_queue,
    int *move_count,
    Neutron *fission_bank,
    int fission_bank_capacity,
    int *fission_bank_count,
    HistoryTallies *history_tallies
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

__global__ void resample_kernel();

__global__ void initialize_source_kernel();

float compute_shannon_entropy(int *bins, int count);

#endif
