#pragma once
#include "hittable.cuh"
class sphere: public hittable {
public:
    __device__ sphere(const point3& center, float radius, material* mat);
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
private:
    point3 center;
    float radius;
    material* mat;
};