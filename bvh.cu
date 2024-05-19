#include <algorithm>
#include "bvh.cuh"
__device__ bool box_compare(const hittable* a, const hittable* b, int axis_index) {
    interval a_axis_interval = a->bounding_box().axis_interval(axis_index);
    interval b_axis_interval = b->bounding_box().axis_interval(axis_index);
    return a_axis_interval.min < b_axis_interval.min;
}
__device__ bool box_x_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 0);
}
__device__ bool box_y_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 1);
}
__device__ bool box_z_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 2);
}

__device__ bvh::bvh(hittable_list* world, curandState* rand_state):bvh(world, rand_state, 0, (world)->last) {

}
__device__ bvh::bvh(hittable_list* world, curandState* rand_state, size_t start, size_t end) {
    int axis = round(cudaRand(rand_state, 0, 2));
    auto comparator = (axis == 0) ? box_x_compare
                                  : (axis == 1) ? box_y_compare
                                                : box_z_compare;
    size_t object_span = end - start;
    if (object_span == 1) {
        left = *((world)->objects + start);
        right = *((world)->objects + start);
    } else if (object_span == 2) {
        left = *((world)->objects + start);
        right = *((world)->objects + start+1);
    } else {
        thrust::sort(((world)->objects + start), ((world)->objects + end), comparator);
        size_t mid = start + object_span/2;
        left = new bvh(world, rand_state, start, mid);
        right = new bvh(world, rand_state, mid, end);
    }
    bbox = aabb(left->bounding_box(), right->bounding_box());
}
__device__ bool bvh::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
    if (!bbox.hit(r, interval{t_min, t_max})) return false;
    bool hit_left = left->hit(r, t_min, t_max, record);
    bool hit_right = right->hit(r, t_min, hit_left ? record.t : t_max, record);
    return hit_left || hit_right;
}
__device__ aabb bvh::bounding_box() const {
    return bbox;
}