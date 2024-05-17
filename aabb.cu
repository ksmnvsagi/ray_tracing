#include "aabb.cuh"
__host__ __device__ interval::interval():min(FLT_MAX), max(FLT_MIN) {

}
__host__ __device__ interval::interval(float min, float max):min(min), max(max) {

}
__host__ __device__ interval::interval(const interval& a, const interval& b) {
    min = a.min <= b.min ? a.min : b.min;
    max = a.max >= b.max ? a.max : b.max;
}
__host__ __device__ interval interval::expand(float d) const {
    return interval{min-d/2, max+d/2};
}
__host__ __device__ float interval::size() const {
    return max-min;
}

__host__ __device__ aabb::aabb() {

}
__host__ __device__ aabb::aabb(const interval& x, const interval& y, const interval& z):x(x), y(y), z(z) {
    if (x.size() < 0.0001f) this->x = x.expand(0.0001f);
    if (y.size() < 0.0001f) this->y = y.expand(0.0001f);
    if (z.size() < 0.0001f) this->z = z.expand(0.0001f);
}
__host__ __device__ aabb::aabb(const point3& a, const point3& b) {
    x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
    y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
    z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
    if (x.size() < 0.0001f) this->x = x.expand(0.0001f);
    if (y.size() < 0.0001f) this->y = y.expand(0.0001f);
    if (z.size() < 0.0001f) this->z = z.expand(0.0001f);
}
__host__ __device__ aabb::aabb(const aabb& box0, const aabb& box1) {
    x = interval(box0.x, box1.x);
    y = interval(box0.y, box1.y);
    z = interval(box0.z, box1.z);
}
__host__ __device__ const interval& aabb::axis_interval(int n) const {
    if (n == 1) return y;
    if (n == 2) return z;
    return x;
}
__host__ __device__ bool aabb::hit(const ray& r, interval ray_t) const {
    const point3& ray_orig = r.orig();
    const vec3& ray_dir  = r.dir();

    for (int axis = 0; axis < 3; axis++) {
        const interval& ax = axis_interval(axis);

        float t0 = (ax.min - ray_orig[axis])/ray_dir[axis];
        float t1 = (ax.max - ray_orig[axis])/ray_dir[axis];

        if (t0 < t1) {
            if (t0 > ray_t.min) ray_t.min = t0;
            if (t1 < ray_t.max) ray_t.max = t1;
        } else {
            if (t1 > ray_t.min) ray_t.min = t1;
            if (t0 < ray_t.max) ray_t.max = t0;
        }

        if (ray_t.max <= ray_t.min) return false;

    }
    return true;
}
