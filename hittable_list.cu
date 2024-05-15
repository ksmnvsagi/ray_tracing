#include "hittable_list.cuh"
__device__ hittable_list::hittable_list(hittable** objects, int size):objects(objects), size(size) {

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