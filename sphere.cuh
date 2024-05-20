#pragma once
#include "hittable.cuh"
class sphere: public hittable {
public:
    __device__ sphere(const point3& center, float radius, material* mat);
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
    __device__ aabb bounding_box() const override;
    __device__ void get_uv(const point3& p, float& u, float& v) const;

    material* mat;
private:
    point3 center;
    float radius;
    aabb bbox;
};