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

__device__ tri::tri(const point3& Q, const vec3& u, const vec3& v, material* mat): Q(Q), u(u), v(v), mat(mat) {
    vec3 n = cross(u, v);
    normal = unit(n);
    D = dot(normal, Q);
    w = n/dot(n,n);
    set_bounding_box();
}
__device__ void tri::set_bounding_box() {
    point3 min_point(
            fminf(fminf(Q.x(), Q.x()+u.x()), Q.x()+v.x()),
            fminf(fminf(Q.y(), Q.y()+u.y()), Q.y()+v.y()),
            fminf(fminf(Q.z(), Q.z()+u.z()), Q.z()+v.z())
    );
    point3 max_point(
            fmaxf(fmaxf(Q.x(), Q.x()+u.x()), Q.x()+v.x()),
            fmaxf(fmaxf(Q.y(), Q.y()+u.y()), Q.y()+v.y()),
            fmaxf(fmaxf(Q.z(), Q.z()+u.z()), Q.z()+v.z())
    );

    bbox = aabb(min_point, max_point);
}
__device__ aabb tri::bounding_box() const {
    return bbox;
}
__device__ bool tri::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
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

__device__ bool tri::is_interior(float a, float b, hit_record& record) const {
    if ((a < 0) || (b < 0) || (a + b > 1)) return false;

    record.u = a;
    record.v = b;
    return true;
}

__device__ hittable_list* create_box(const point3& a, const point3& b, material* mat) {
    hittable_list* sides = new hittable_list(6);
    point3 min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    point3 max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));
    vec3 dx = vec3(max.x() - min.x(), 0, 0);
    vec3 dy = vec3(0, max.y() - min.y(), 0);
    vec3 dz = vec3(0, 0, max.z() - min.z());

    sides->add(new quad(point3(min.x(), min.y(), max.z()),  dx,  dy, mat)); // front
    sides->add(new quad(point3(max.x(), min.y(), max.z()), -dz,  dy, mat)); // right
    sides->add(new quad(point3(max.x(), min.y(), min.z()), -dx,  dy, mat)); // back
    sides->add(new quad(point3(min.x(), min.y(), min.z()),  dz,  dy, mat)); // left
    sides->add(new quad(point3(min.x(), max.y(), max.z()),  dx, -dz, mat)); // top
    sides->add(new quad(point3(min.x(), min.y(), min.z()),  dx,  dz, mat)); // bottom

    return sides;
}
__device__ bvh* create_object(vec3* vertices, vec3* triangles, int total_triangles, material* mat) {
    hittable_list* sides = new hittable_list(total_triangles);
    for (size_t i = 0; i < total_triangles; i++) {
        int index0 = (int)triangles[i].x();
        int index1 = (int)triangles[i].y();
        int index2 = (int)triangles[i].z();

        point3 v0 = vertices[index0];
        point3 v1 = vertices[index1];
        point3 v2 = vertices[index2];
        vec3 dir1 = v1 - v0;
        vec3 dir2 = v2 - v0;
        sides->add(new tri(v0, dir1, dir2, mat));
    }
    bvh* node = new bvh(sides);
    return node;
}
__device__ hittable_list* create_complex(const point3& min_corner, const point3& max_corner, material* mat) {
    hittable_list* sides = new hittable_list(12);

    point3 v0(min_corner.x(), min_corner.y(), min_corner.z()); // 0 0 0
    point3 v1(max_corner.x(), min_corner.y(), min_corner.z()); // 5 0 0
    point3 v2(max_corner.x(), max_corner.y(), min_corner.z()); // 5 5 0
    point3 v3(min_corner.x(), max_corner.y(), min_corner.z()); // 0 5 0
    point3 v4(min_corner.x(), min_corner.y(), max_corner.z()); // 0 0 5
    point3 v5(max_corner.x(), min_corner.y(), max_corner.z()); // 5 0 5
    point3 v6(max_corner.x(), max_corner.y(), max_corner.z()); // 5 5 5
    point3 v7(min_corner.x(), max_corner.y(), max_corner.z()); // 0 5 5

    // Front face
    sides->add(new tri(v0, v1-v0, v2-v0, mat));
    sides->add(new tri(v0, v2-v0, v3-v0, mat));
    // Right face
    sides->add(new tri(v1, v5-v1, v6-v1, mat));
    sides->add(new tri(v1, v6-v1, v2-v1, mat));
    // Back face
    sides->add(new tri(v5, v4-v5, v7-v5, mat));
    sides->add(new tri(v5, v7-v5, v6-v5, mat));
    // Left face
    sides->add(new tri(v4, v0-v4, v3-v4, mat));
    sides->add(new tri(v4, v3-v4, v7-v4, mat));
    // Top face
    sides->add(new tri(v3, v2-v3, v6-v3, mat));
    sides->add(new tri(v3, v6-v3, v7-v3, mat));
    // Bottom face
    sides->add(new tri(v0, v4-v0, v5-v0, mat));
    sides->add(new tri(v0, v5-v0, v1-v0, mat));

    return sides;
}
