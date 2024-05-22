#pragma once
#include "hittable.cuh"

class hittable_list: public hittable {
public:
    __device__ hittable_list(int size);
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
    __device__ aabb bounding_box() const override;
    __device__ void add(hittable* object);

    int last;
    hittable** objects; // list of pointers to objects (due to polymorphism)
private:
    int size;
    aabb bbox;
};
