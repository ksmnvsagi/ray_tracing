#pragma once
#include "config.cuh"
#include "hittable.cuh"
#include "ray.cuh"
class camera {
public:
    float aspect_ratio;
    int image_height;
    int image_width;
    point3 center;
    point3 pixel00;
    vec3 delta_x;       // distance to next pixel in x-direction
    vec3 delta_y;       // distance to next pixel in y-direction
    int num_samples;
    float vfov;         // vertical field of view
    point3 lookat;
    vec3 vup;           // camera-relative up vector
    vec3 u;             // vector pointing right from camera
    vec3 v;             // vector pointing up from camera
    vec3 w;             // vector pointing away from lookat point

    __device__ __host__ camera(float aspect_ratio, int image_width, point3 center, point3 lookat,
                               float vfov = 90.0f, int num_samples = 100);
    __device__ color ray_color(const ray& r, hittable** world, curandState* rand_state) const;
};