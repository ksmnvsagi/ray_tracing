#include "ray.cuh"
__device__ __host__ ray::ray():origin(0, 0, 0), direction(0, 0, 0) {

}
__device__ __host__ ray::ray(const point3& origin, const vec3& direction):origin(origin), direction(direction) {

}
__device__ __host__ const point3& ray::orig() const {
    return origin;
}
__device__ __host__ const vec3& ray::dir() const {
    return direction;
}
__device__ __host__ point3 ray::at(float t) const {
    return origin + t*direction;
}