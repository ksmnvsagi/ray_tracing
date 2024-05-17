#include "quad.cuh"
__device__ quad::quad(const point3& Q, const vec3& u, const vec3& v, material* mat): Q(Q), u(u), v(v), mat(mat) {
    vec3 n = cross(u, v);
    normal = unit(n);
    D = dot(normal, Q);
    w = n/dot(n,n);
    set_bounding_box();
}
__device__ void quad::set_bounding_box() {
    aabb quad_diag1 = aabb(Q, Q + u + v);
    aabb quad_diag2 = aabb(Q + u, Q + v);
    bbox = aabb(quad_diag1, quad_diag2);
}
__device__ aabb quad::bounding_box() const {
    return bbox;
}
__device__ bool quad::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
    float denominator = dot(normal, r.dir());
    if (fabs(denominator) < FLT_MIN) return false; // parallel ray
    float t = (D - dot(normal, r.orig()))/denominator;
    if (t_min > t || t_max < t) return false; // ray already hit something else
    point3 intersection = r.at(t);
    vec3 q_to_intersect = intersection - Q;
    float alpha = dot(w, cross(q_to_intersect, v));
    float beta = dot(w, cross(u, q_to_intersect));

    if (!is_interior(alpha, beta, record)) return false;
    record.t = t;
    record.p = intersection;
    record.mat = mat;
    record.front_face = dot(r.dir(), normal) < 0;
    record.normal = record.front_face ? normal : -normal;
    return true;
}
__device__ bool quad::is_interior(float a, float b, hit_record& record) const {
    if ((0.f > a || 1.f < a) || (0.f > b || 1.f < b)) return false;

    record.u = a;
    record.v = b;
    return true;
}