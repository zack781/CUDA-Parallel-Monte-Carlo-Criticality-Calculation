#include <stdio.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/random.h>

__global__ void dkernel() {
    printf("Hello world\n");
}

struct random_point {
    __device__
    float2 operator()(unsigned int idx) const {
        thrust::default_random_engine rng(idx);
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return make_float2(dist(rng), dist(rng));
    }
};

struct inside_circle {
    __device__
    unsigned int operator()(float2 p) const {
        return (((p.x-0.5)*(p.x-0.5)+(p.y-0.5)*(p.y-0.5))<0.25) ? 1 : 0;
    }
};

int main() {
    dkernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    size_t N = 5000000; // Number of Monte-Carlo simulations

    // DEVICE: Generate random points within an unit square
    thrust::device_vector<float2> d_random(N);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_random.begin(), random_point());

    // DEVICE: Flags to mark points as lying inside or outside the circle
    thrust::device_vector<unsigned int> d_inside(N);

    // DEVIVE: Function evaluation. Mark points as inside or outside
    thrust::transform(d_random.begin(), d_random.end(),
                      d_inside.begin(), inside_circle());

    // DEVICE: Aggregation
    size_t total = thrust::count(d_inside.begin(), d_inside.end(), 1);

    // HOST: Print estimate of PI
    std::cout << "PI: " << 4.0 * (float)total/(float)N << std::endl;

    return 0;
}
