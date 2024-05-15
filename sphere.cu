#include "sphere.cuh"
__device__ sphere::sphere(const point3& center, float radius, material* mat):center(center), radius(radius), mat(mat) {

}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
    vec3 oc = center - r.orig();
    float a = r.dir().squared_length();
    float h = dot(r.dir(), oc);
    float c = oc.squared_length() - radius*radius;
    float discriminant = h*h - a*c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);
    float root = (h-sqrtd)/a;
    if (root <= t_min || t_max <= root) {
        root = (h+sqrtd) / a;
        if (root <= t_min || t_max <= root) return false;
    }
    record.t = root;
    record.p = r.at(record.t);
    record.normal = (record.p - center) / radius;
    record.front_face = dot(r.dir(), record.normal) < 0;
    record.mat = mat;
    return true;
}