#pragma once
#include "hittable.cuh"
#include "hittable_list.cuh"
#include "quad.cuh"
#include "bvh.cuh"

class quad: public hittable {
public:
    __device__ quad(const point3& Q, const vec3& u, const vec3& v, material* mat);
    __device__ void set_bounding_box();
    __device__ aabb bounding_box() const override;
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
    __device__ bool is_interior(float a, float b, hit_record& record) const;
    material* mat;
    point3 Q;
    vec3 u, v;
    aabb bbox;
    vec3 normal;
    float D;
    vec3 w;
};

class tri: public hittable {
public:
    __device__ tri(const point3& Q, const vec3& u, const vec3& v, material* mat);
    __device__ void set_bounding_box();
    __device__ aabb bounding_box() const override;
    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& record) const override;
    __device__ bool is_interior(float a, float b, hit_record& record) const;
    material* mat;
    point3 Q;
    vec3 u, v;
    aabb bbox;
    vec3 normal;
    float D;
    vec3 w;
};

__device__ hittable_list* create_box(const point3& a, const point3& b, material* mat);
__device__ bvh* create_object(vec3* vertices, vec3* triangles, int total_triangles, material* mat);
__device__ hittable_list* create_complex(const point3& min_corner, const point3& max_corner, material* mat);

