#pragma once
#include "ray.cuh"
#include "config.cuh"
#include "material.cuh"

class material;

struct hit_record {
    point3 p;           // point of intersection
    vec3 normal;        // normal to point of intersection
    float t;            // t-value at which ray intersects
    bool front_face;
    material* mat;
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& record) const = 0;
};