#pragma once
#include "hittable.cuh"

class quad: public hittable {
public:
    __device__ quad(const point3& Q, const vec3& u, const vec3& v, material* mat);
    __device__ void set_bounding_box();
    __device__ aabb bounding_box() const override;
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
    __device__ bool is_interior(float a, float b, hit_record& record) const;

    material* mat;
private:
    point3 Q;
    vec3 u, v;
    aabb bbox;
    vec3 normal;
    float D;
    vec3 w;
};