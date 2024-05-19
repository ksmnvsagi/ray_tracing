#pragma once

#include "hittable.cuh"
#include "hittable_list.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <curand_kernel.h>

__device__ bool box_compare(const hittable* a, const hittable* b);
__device__ bool box_x_compare(const hittable* a, const hittable* b);
__device__ bool box_y_compare(const hittable* a, const hittable* b);
__device__ bool box_z_compare(const hittable* a, const hittable* b);

class bvh: public hittable {
public:
    __device__ bvh(hittable_list* world, curandState* rand_state);
    __device__ bvh(hittable_list* world, curandState* rand_state, size_t start, size_t end);
    __device__ ~bvh();
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
    __device__ aabb bounding_box() const override;
private:
    hittable* left;     // left bvh node
    hittable* right;    // right bvh node
    aabb bbox;
};
