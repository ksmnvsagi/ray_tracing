#pragma once
#include "hittable_list.cuh"
#include "bvh.cuh"
#include "sphere.cuh"
__device__ void cornell(int size, bvh** node);
__device__ void tri_test(int size, bvh** node);
__device__ void obj_test(int size, bvh** node, vec3* vertices, vec3* triangles, int total_triangles);