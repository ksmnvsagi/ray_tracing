#pragma once
#include "hittable.cuh"

class hittable_list: public hittable {
public:
    __device__ hittable_list(hittable** objects, int size);
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
private:
    hittable** objects;
    int size;
};
