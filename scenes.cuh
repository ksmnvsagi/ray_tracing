#pragma once
#include "hittable_list.cuh"
#include "bvh.cuh"
#include "sphere.cuh"
__device__ void cornell(int size, bvh** node, image* img, curandState* rand_state);
