#pragma once
#include "hittable_list.cuh"
class translate: public hittable {
public:
    __device__ translate(hittable* object, vec3 offset);
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
    __device__ aabb bounding_box() const override;
private:
    hittable* object;
    vec3 offset;
    aabb bbox;
};

class rotate_y: public hittable {
public:
    __device__ rotate_y(hittable* object, float angle);
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
    __device__ aabb bounding_box() const override;
private:
    hittable* object;
    vec3 offset;
    aabb bbox;
    float sin_theta; // angle between original ray and new
    float cos_theta;
};



