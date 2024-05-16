#pragma once
#include "ray.cuh"

class texture {
public:
    __device__ __host__ virtual color value(float u, float v, const point3& p) const = 0;
};

class solid_color: public texture {
public:
    __device__ __host__ solid_color(const color& albedo);
    __device__ __host__ solid_color(float r, float g, float b);
    __device__ __host__ color value(float u, float v, const point3& p) const override;
private:
    color albedo;
};

class checker_texture: public texture {
public:
    __device__ __host__ checker_texture(float scale, texture* even, texture* odd);
    __device__ __host__ checker_texture(float scale, const color& c1, const color& c2);
    __device__ __host__ color value(float u, float v, const point3& p) const override;
private:
    float inv_scale;
    texture* even;
    texture* odd;
};