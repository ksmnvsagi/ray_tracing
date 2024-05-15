#pragma once
#include "config.cuh"
#include "hittable.cuh"

struct hit_record;

class material {
public:
    __device__ virtual bool scatter(const ray& r, const hit_record& record, color& attenuation,
                                    ray& scattered, curandState* rand_state) const = 0;
};

class lambertian: public material {
public:
    __device__ lambertian(const color& albedo);
    __device__ bool scatter(const ray& r, const hit_record& record, color& attenuation,
                            ray& scattered, curandState* rand_state) const override;
private:
    color albedo;
};

class metal: public material {
public:
    __device__ metal(const color& albedo, float fuzz);
    __device__ bool scatter(const ray& r, const hit_record& record, color& attenuation,
                            ray& scattered, curandState* rand_state) const override;
private:
    color albedo;
    float fuzz;
};

class dielectric: public material {
public:
    __device__ dielectric(float refraction_index);

    __device__ bool scatter(const ray& r, const hit_record& record, color& attenuation,
                            ray& scattered, curandState* rand_state) const override;
private:
    float refraction_index;
    __device__ float schlick(float cosine, float ri) const;
};