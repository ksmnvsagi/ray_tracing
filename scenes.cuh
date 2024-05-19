#pragma once
#include "hittable_list.cuh"
#include "bvh.cuh"
#include "sphere.cuh"
__device__ void global(int size, bvh** node, curandState* rand_state);
__device__ void checker_spheres(int size, bvh** node, curandState* rand_state);
__device__ void earth(int size, bvh** node, image* img, curandState* rand_state);
__device__ void quads(int size, bvh** node, curandState* rand_state);
__device__ void empty_cornell(int size, bvh** node, image* img, curandState* rand_state);
