#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <memory>
#include <typeinfo>
#define _USE_MATH_DEFINES
#include <cmath>
#include <curand_kernel.h>

// convert from linear space to gamma
#define GAMMA(val) (val>0 ? sqrt(val) : 0)
#define clamp(val) (val < 0.f ? 0 : (val > 1.f ? 1.f : val))

#define cudaCheck(val) checkCuda( (val), #val, __FILE__, __LINE__ )
inline void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error: " << cudaGetErrorString( result ) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        exit(1);
    }
}

__device__ inline float cudaRand(curandState* rand_state, float min = 0.f, float max = 1.f) {
    return min + (max-min)*curand_uniform(rand_state);
}