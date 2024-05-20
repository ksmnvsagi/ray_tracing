#pragma once
#include "config.cuh"
#include "hittable.cuh"
#include "texture.cuh"

struct hit_record;

class material {
public:
    __device__ virtual color emit(float u, float v, const point3& p) const = 0;
    __device__ virtual bool scatter(const ray& r, const hit_record& record, color& attenuation,
                                    ray& scattered, curandState* rand_state) const = 0;
};

class lambertian: public material {
public:
    __device__ lambertian(const color& albedo);
    __device__ lambertian(texture* tex);
    __device__ bool scatter(const ray& r, const hit_record& record, color& attenuation,
                            ray& scattered, curandState* rand_state) const override;
    __device__ color emit(float u, float v, const point3& p) const override;
private:
    color albedo;
    texture* tex;
};

class metal: public material {
public:
    __device__ metal(const color& albedo, float fuzz);
    __device__ bool scatter(const ray& r, const hit_record& record, color& attenuation,
                            ray& scattered, curandState* rand_state) const override;
    __device__ color emit(float u, float v, const point3& p) const override;
private:
    color albedo;
    float fuzz;
};

class dielectric: public material {
public:
    __device__ dielectric(float refraction_index);

    __device__ bool scatter(const ray& r, const hit_record& record, color& attenuation,
                            ray& scattered, curandState* rand_state) const override;
    __device__ color emit(float u, float v, const point3& p) const override;
private:
    float refraction_index;
    __device__ float schlick(float cosine, float ri) const;
};

class diffuse_light: public material {
public:
    __device__ diffuse_light(const color& emitted_color);
    __device__ diffuse_light(texture* tex);
    __device__ color emit(float u, float v, const point3& p) const override;
    __device__ bool scatter(const ray& r, const hit_record& record, color& attenuation,
                                           ray& scattered, curandState* rand_state) const override;
private:
    texture* tex;
};