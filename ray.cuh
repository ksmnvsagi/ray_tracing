#pragma once
#include "config.cuh"
#include "vec3.cuh"

class ray {
public:
    __device__ __host__ ray();
    __device__ __host__ ray(const point3& origin, const vec3& direction);
    __device__ __host__ const point3& orig() const;
    __device__ __host__ const vec3& dir() const;
    __device__ __host__ point3 at(float t) const;
private:
    point3 origin;
    vec3 direction;
};