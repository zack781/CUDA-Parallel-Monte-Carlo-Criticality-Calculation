// CPU port of the CUDA Monte Carlo criticality simulation.
// Collapses move_kernel + collision_kernel into a single per-neutron history
// loop, matching CUDA physics exactly (same XS, same scatter formula, same
// group lookup, same periodic boundary).
//
// Compile locally:  g++ -O3 -std=c++17 -o cpu_sim src/cpu_sim.cpp
// Compile on Perlmutter: CC -O3 -std=c++17 -o cpu_sim src/cpu_sim.cpp
// Run: ./cpu_sim [neutrons] [generations] [boron_ppm]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cstdint>
#include <climits>
#include <cfloat>

static constexpr float PI         = 3.14159265358979323846f;
static constexpr int   NUM_GROUPS  = 10;
static constexpr int   NUM_REGIONS = 3;

// ── Geometry ──────────────────────────────────────────────────────────────────
static constexpr float R_FUEL     = 0.53f;
static constexpr float R_CLAD_OUT = 0.90f;
static constexpr float HALF_PITCH = 1.837f * 0.5f;

enum Region { FUEL = 0, CLAD = 1, MODERATOR = 2 };

// ── Cross-section tables (mirror of sim.cuh) ──────────────────────────────────
// Energy group lower bounds in MeV, fast→thermal.
static const float GROUP_ENERGY[NUM_GROUPS] = {
    3.0e+1f, 3.0e+0f, 3.0e-1f, 3.0e-2f, 3.0e-3f,
    3.0e-4f, 3.0e-5f, 3.0e-6f, 3.0e-7f, 3.0e-8f
};

// [group][region]  (non-const so boron can be added at runtime)
static float sigma_f[NUM_GROUPS][NUM_REGIONS] = {
    {1.05e-1f, 0.0f, 0.0f}, {5.96e-2f, 0.0f, 0.0f},
    {6.02e-2f, 0.0f, 0.0f}, {1.06e-1f, 0.0f, 0.0f},
    {2.46e-1f, 0.0f, 0.0f}, {2.50e-1f, 0.0f, 0.0f},
    {1.07e-1f, 0.0f, 0.0f}, {1.28e+0f, 0.0f, 0.0f},
    {9.30e+0f, 0.0f, 0.0f}, {2.58e+1f, 0.0f, 0.0f}
};

static float sigma_c[NUM_GROUPS][NUM_REGIONS] = {
    {1.41e-6f, 1.71e-2f, 3.34e-6f}, {1.34e-3f, 7.83e-3f, 3.34e-6f},
    {1.10e-2f, 2.83e-4f, 2.56e-7f}, {3.29e-2f, 4.52e-6f, 6.63e-7f},
    {8.23e-2f, 1.06e-5f, 2.24e-7f}, {4.28e-2f, 4.39e-6f, 1.27e-7f},
    {9.90e-2f, 1.25e-5f, 2.02e-7f}, {2.51e-1f, 3.98e-5f, 6.02e-7f},
    {2.12e+0f, 1.26e-4f, 1.84e-6f}, {4.30e+0f, 3.95e-4f, 5.76e-6f}
};

static float sigma_s[NUM_GROUPS][NUM_REGIONS] = {
    {2.76e-1f, 1.44e-1f, 1.27e-2f}, {3.88e-1f, 1.76e-1f, 7.36e-2f},
    {4.77e-1f, 3.44e-1f, 2.65e-1f}, {6.88e-1f, 2.66e-1f, 5.72e-1f},
    {9.38e-1f, 2.06e-1f, 6.69e-1f}, {1.52e+0f, 2.14e-1f, 6.81e-1f},
    {2.30e+0f, 2.23e-1f, 6.82e-1f}, {2.45e+0f, 2.31e-1f, 6.83e-1f},
    {9.79e+0f, 2.40e-1f, 6.86e-1f}, {4.36e+1f, 2.41e-1f, 6.91e-1f}
};

// B-10 microscopic (n,α) XS in barns per group, 1/v from 3840 b at 0.025 eV.
static const float SIGMA_B10_MICRO[NUM_GROUPS] = {
    0.11f, 0.35f, 1.1f, 3.5f, 11.1f, 35.1f, 111.0f, 350.0f, 1109.0f, 3507.0f
};

// Adds B-10 to the moderator capture XS before the generation loop.
static void apply_boron(float ppm) {
    float N_B10 = ppm * 6.022e16f; // atoms/cm³ per ppm (pure B-10 in water)
    for (int g = 0; g < NUM_GROUPS; g++)
        sigma_c[g][MODERATOR] += N_B10 * SIGMA_B10_MICRO[g] * 1e-24f;
}

// ── Cross-section lookup ──────────────────────────────────────────────────────
struct XS { float sig_f, sig_c, sig_s, sig_t; };

// Same counting logic as transport.cu::get_energy_group.
static int get_group(float energy) {
    int g = 0;
    for (int t = 0; t < NUM_GROUPS - 1; t++)
        g += (energy < GROUP_ENERGY[t]);
    return g;
}

static XS get_xs(float energy, int region) {
    int g = get_group(energy);
    XS xs;
    xs.sig_f = sigma_f[g][region];
    xs.sig_c = sigma_c[g][region];
    xs.sig_s = sigma_s[g][region];
    xs.sig_t = xs.sig_f + xs.sig_c + xs.sig_s;
    return xs;
}

// ── RNG (one mt19937 per simulation; seed matches CUDA default 1234) ──────────
struct RNG {
    std::mt19937 eng;
    std::uniform_real_distribution<float> udist{0.0f, 1.0f};
    std::normal_distribution<float>       ndist{0.0f, 1.0f};

    explicit RNG(uint64_t seed = 1234) : eng(seed) {}

    float uniform() { return udist(eng); }
    float normal()  { return ndist(eng); }

    // Maxwell speed distribution: same as rng.cu::sample_initial_energy.
    float initial_energy() {
        float a = normal(), b = normal(), c = normal();
        return std::sqrt(a*a + b*b + c*c);
    }

    void isotropic_dir(float &ux, float &uy) {
        float theta = 2.0f * PI * uniform();
        ux = std::cos(theta);
        uy = std::sin(theta);
    }

    // Same as rng.cu::sample_fission_multiplicity.
    int fission_nu() { return uniform() < 0.5f ? 2 : 3; }
};

// ── Geometry helpers ──────────────────────────────────────────────────────────

// Shortest positive distance to a circle of radius r; FLT_MAX if none.
static float circle_dist(float x, float y, float ux, float uy, float r) {
    float b    = 2.0f * (x*ux + y*uy);
    float c    = x*x + y*y - r*r;
    float disc = b*b - 4.0f*c;
    if (disc < 0.0f) return FLT_MAX;
    float sq = std::sqrt(disc);
    constexpr float eps = 1e-9f;
    float t1 = (-b - sq) * 0.5f;
    if (t1 > eps) return t1;
    float t2 = (-b + sq) * 0.5f;
    return t2 > eps ? t2 : FLT_MAX;
}

// Positive distance to a flat wall; FLT_MAX if not heading toward it.
static float flat_dist(float pos, float dir, float wall) {
    if (std::abs(dir) < 1e-15f) return FLT_MAX;
    float t = (wall - pos) / dir;
    return t > 1e-9f ? t : FLT_MAX;
}

// ── Tallies ───────────────────────────────────────────────────────────────────
struct Tallies {
    long long fission = 0, capture = 0, scatter = 0;
    long long leakage = 0, neutrons_produced = 0, lost = 0;

    void operator+=(const Tallies &o) {
        fission += o.fission; capture += o.capture; scatter += o.scatter;
        leakage += o.leakage; neutrons_produced += o.neutrons_produced;
        lost    += o.lost;
    }
};

// ── Fission site (position only; energy is resampled at generation start) ─────
struct Site { float x, y; };

// ── Per-neutron history ───────────────────────────────────────────────────────
// Collapses move_kernel + collision_kernel into a sequential history loop.
// need_dir mirrors regionchange==0 in the CUDA code: direction is sampled
// at the start of each free-flight segment (after scatter or at generation
// start), not at boundary crossings.
static void transport_one(
    float x, float y, float energy, int region,
    RNG &rng,
    std::vector<Site> &fission_bank,
    Tallies &t
) {
    bool need_dir = true;  // regionchange==0 in CUDA
    float ux = 1.0f, uy = 0.0f;

    for (;;) {
        // ── Sample direction for new free-flight segment ─────────────────────
        if (need_dir) {
            rng.isotropic_dir(ux, uy);
            need_dir = false;
        }

        XS xs = get_xs(energy, region);
        if (xs.sig_t <= 0.0f) { t.lost++; return; }

        float d = -std::log(rng.uniform()) / xs.sig_t;

        // ── Nearest boundary ─────────────────────────────────────────────────
        float dmin        = FLT_MAX;
        int   next_region = region;
        bool  periodic    = false;
        int   paxis       = 0;       // periodic axis: 0=x, 1=y

        if (region == FUEL) {
            float df = circle_dist(x, y, ux, uy, R_FUEL);
            if (df < dmin) { dmin = df; next_region = CLAD; }

        } else if (region == CLAD) {
            float df = circle_dist(x, y, ux, uy, R_FUEL);
            float dc = circle_dist(x, y, ux, uy, R_CLAD_OUT);
            if (df <= dc) { dmin = df; next_region = FUEL; }
            else          { dmin = dc; next_region = MODERATOR; }

        } else { // MODERATOR
            float dc = circle_dist(x, y, ux, uy, R_CLAD_OUT);
            if (dc < dmin) { dmin = dc; next_region = CLAD; }

            // Four flat cell walls (periodic).
            auto try_wall = [&](float pos, float dir, float wall, int axis) {
                float t = flat_dist(pos, dir, wall);
                if (t < dmin) { dmin = t; periodic = true; paxis = axis; }
            };
            if (ux > 0.0f) try_wall(x, ux,  HALF_PITCH, 0);
            if (ux < 0.0f) try_wall(x, ux, -HALF_PITCH, 0);
            if (uy > 0.0f) try_wall(y, uy,  HALF_PITCH, 1);
            if (uy < 0.0f) try_wall(y, uy, -HALF_PITCH, 1);
        }

        if (dmin == FLT_MAX) { t.lost++; return; }

        // ── Move ─────────────────────────────────────────────────────────────
        if (d >= dmin) {
            // Boundary crossing.
            x += dmin * ux;
            y += dmin * uy;

            if (periodic) {
                // Periodic wrap: mirrors CUDA's  x = -x  /  y = -y.
                if (paxis == 0) x = -x; else y = -y;
                region = MODERATOR;
                t.leakage++;        // matches CUDA tally name (cosmetic)
                // direction is preserved — same as regionchange=1 in CUDA
            } else {
                // Nudge past the surface to avoid re-detecting it.
                constexpr float eps = 1e-5f;
                x += eps * ux;
                y += eps * uy;
                region = next_region;
                // direction preserved through region boundary
            }

        } else {
            // Collision.
            x += d * ux;
            y += d * uy;

            float xi  = rng.uniform();
            float p_f = xs.sig_f / xs.sig_t;
            float p_c = xs.sig_c / xs.sig_t;

            if (xi <= p_f) {
                // Fission: kill neutron, record ν daughter sites.
                t.fission++;
                int nu = rng.fission_nu();
                t.neutrons_produced += nu;
                for (int j = 0; j < nu; j++)
                    fission_bank.push_back({x, y});
                return;

            } else if (xi <= p_f + p_c) {
                // Capture.
                t.capture++;
                return;

            } else {
                // Scatter: deterministic energy loss (matches transport.cu).
                t.scatter++;
                float A     = (region == FUEL) ? 238.02891f :
                              (region == CLAD) ?  26.981539f : 4.5f;
                float ratio = (A - 1.0f) / (A + 1.0f);
                float ksi   = 1.0f + std::log(ratio) * (A-1.0f)*(A-1.0f) / (2.0f*A);
                energy *= std::exp(-ksi);
                need_dir = true;    // regionchange=0: sample new direction
            }
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    int   N               = argc > 1 ? std::atoi(argv[1]) : 10000;
    int   num_generations = argc > 2 ? std::atoi(argv[2]) : 20;
    float boron_ppm       = argc > 3 ? std::atof(argv[3]) : 0.0f;

    if (N <= 0 || num_generations <= 0) {
        std::fprintf(stderr, "Usage: %s [neutrons] [generations] [boron_ppm]\n", argv[0]);
        return 1;
    }

    if (boron_ppm > 0.0f) {
        apply_boron(boron_ppm);
        std::printf("Boron-10: %.1f ppm applied to moderator capture XS\n", boron_ppm);
    }
    std::printf("Config: neutrons=%d  generations=%d  boron_ppm=%.1f\n\n",
                N, num_generations, boron_ppm);

    RNG rng(1234);

    // Initial source: uniform in fuel disk (matches initialize_neutrons).
    std::vector<Site> source(N);
    for (auto &s : source) {
        float theta = 2.0f * PI * rng.uniform();
        float r     = R_FUEL * std::sqrt(rng.uniform());
        s = {r * std::cos(theta), r * std::sin(theta)};
    }

    Tallies total{};
    int completed = 0;

    for (int gen = 0; gen < num_generations; gen++) {
        std::vector<Site> fission_bank;
        fission_bank.reserve(N * 3);
        Tallies g{};

        for (int i = 0; i < N; i++) {
            float energy = rng.initial_energy();
            transport_one(source[i].x, source[i].y, energy, FUEL,
                          rng, fission_bank, g);
        }

        double keff = (double)fission_bank.size() / N;
        std::printf("Generation %3d  Fission Bank Sites = %6d  k_eff = %.6f\n",
                    gen, (int)fission_bank.size(), keff);
        completed++;

        if (fission_bank.empty()) break;

        // Resample source for next generation (matches resample_kernel).
        source.resize(N);
        for (int i = 0; i < N; i++) {
            int idx = static_cast<int>(rng.uniform() * fission_bank.size());
            if (idx >= (int)fission_bank.size()) idx = (int)fission_bank.size() - 1;
            source[i] = fission_bank[idx];
        }

        total += g;
    }

    long long interactions = total.scatter + total.capture + total.fission;
    long long absorption   = total.capture + total.fission;
    double    avg_nu       = total.fission > 0
                           ? (double)total.neutrons_produced / total.fission : 0.0;

    std::printf("\n");
    std::printf("Number of Neutrons.......................=  %d\n",    N);
    std::printf("Completed Generations....................=  %d\n",    completed);
    std::printf("Number of Interactions...................=  %lld\n",  interactions);
    std::printf("Number of Scattering Events..............=  %lld\n", total.scatter);
    std::printf("Number of Capture Events.................=  %lld\n", total.capture);
    std::printf("Number of Fission Events.................=  %lld\n", total.fission);
    std::printf("Number of Absorption Events..............=  %lld\n", absorption);
    std::printf("Average nu...............................=  %.6f\n",  avg_nu);
    std::printf("Number of Neutrons Produced by Fission...=  %lld\n", total.neutrons_produced);
    std::printf("Number of Neutrons Leaked from System....=  %lld\n", total.leakage);
    std::printf("Lost (no surface found)..................=  %lld\n",  total.lost);

    return 0;
}
