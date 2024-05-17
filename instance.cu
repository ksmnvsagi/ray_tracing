#include "instance.cuh"
__device__ translate::translate(hittable* object, vec3 offset): object(object), offset(offset) {
    bbox = object->bounding_box() + offset;
}
__device__ bool translate::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
    ray offset_r(r.orig() - offset, r.dir());

    if (!object->hit(offset_r, t_min, t_max, record)) return false;
    record.p += offset;
    return true;
}
__device__ aabb translate::bounding_box() const {
    return bbox;
}

__device__ rotate_y::rotate_y(hittable* object, float angle): object(object), offset(offset) {
    float radians = float(angle)*M_PI/180;
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    bbox = object->bounding_box();
    point3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    point3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float x = i*bbox.axis_interval(0).max + (1-i)*bbox.axis_interval(0).min;
                float y = j*bbox.axis_interval(1).max + (1-j)*bbox.axis_interval(1).min;
                float z = k*bbox.axis_interval(2).max + (1-k)*bbox.axis_interval(2).min;

                float newx = cos_theta*x + sin_theta*z;
                float newz = -sin_theta*x + cos_theta*z;

                vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
}
__device__ bool rotate_y::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
    point3 origin = r.orig();
    vec3 direction = r.dir();
    origin[0] = cos_theta*r.orig()[0] - sin_theta*r.orig()[2];
    origin[2] = sin_theta*r.orig()[0] + cos_theta*r.orig()[2];
    direction[0] = cos_theta*r.dir()[0] - sin_theta*r.dir()[2];
    direction[2] = sin_theta*r.dir()[0] + cos_theta*r.dir()[2];
    ray rotated_r(origin, direction);

    if (!object->hit(rotated_r, t_min, t_max, record)) return false;

    point3 p = record.p;
    p[0] = cos_theta*record.p[0] + sin_theta*record.p[2];
    p[2] = -sin_theta*record.p[0] + cos_theta*record.p[2];

    vec3 normal = record.normal;
    normal[0] = cos_theta*record.normal[0] + sin_theta*record.normal[2];
    normal[2] = -sin_theta*record.normal[0] + cos_theta*record.normal[2];

    record.p = p;
    record.normal = normal;
    return true;
}
__device__ aabb rotate_y::bounding_box() const {
    return bbox;
}