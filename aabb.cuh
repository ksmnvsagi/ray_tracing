#pragma once
#include "vec3.cuh"
#include "ray.cuh"

struct interval {
    float min, max;
    __host__ __device__ interval();
    __host__ __device__ interval(float min, float max);
    __host__ __device__ interval(const interval& a, const interval& b);
    __host__ __device__ interval expand(float d) const;
    __host__ __device__ float size() const;
};
__host__ __device__ interval operator+(const interval& inter, float displace);
__host__ __device__ interval operator+(float displace, const interval& inter);

class aabb {
public:
    __host__ __device__ aabb();
    __host__ __device__ aabb(const interval& x, const interval& y, const interval& z);
    __host__ __device__ aabb(const point3& a, const point3& b);
    __host__ __device__ aabb(const aabb& box0, const aabb& box1);
    __host__ __device__ const interval& axis_interval(int n) const;
    __host__ __device__ bool hit(const ray& r, interval ray_t) const;
private:
    interval x, y, z;
};

__host__ __device__ aabb operator+(const aabb& bbox, const vec3& offset);
__host__ __device__ aabb operator+(const vec3& offset, const aabb& bbox);