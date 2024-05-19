#include "hittable_list.cuh"
__device__ hittable_list::hittable_list(int size):size(size), last(0) {
    objects = (hittable**)malloc(size);
}
__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
    hit_record temp_record;
    bool hit_any = false;
    float closest_dist = t_max;
    for (int i=0; i<size; i++) {
        if (objects[i]->hit(r, t_min, closest_dist, temp_record)) {
            hit_any = true;
            closest_dist = temp_record.t;
            record = temp_record;
        }
    }
    return hit_any;
}
__device__ aabb hittable_list::bounding_box() const {
    return bbox;
}
__device__ void hittable_list::add(hittable* object) {
    if (last==size) return;
    *(objects + (last++)) = object;
    bbox = aabb(bbox, object->bounding_box());
}