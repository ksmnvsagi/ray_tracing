#include "camera.cuh"
#include "hittable_list.cuh"

__device__ __host__ camera::camera(float aspect_ratio, int image_width, point3 center, point3 lookat,
                                   float vfov, int num_samples):
        aspect_ratio(aspect_ratio), image_width(image_width), center(center), lookat(lookat),
        vfov(vfov), num_samples(num_samples) {
    // right-handed rule, negative z points towards the image
    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    vup = vec3(0,1,0);

    float focal_length = (center - lookat).length(); // distance from camera to the viewport
    float theta = vfov*(float)M_PI/180.0f;
    float h = tan(theta/2);
    float viewport_height = 2*h*focal_length;
    float viewport_width = viewport_height * (float(image_width)/image_height);
    w = unit(center - lookat);
    u = unit(cross(vup, w));
    v = cross(w, u);

    vec3 viewport_x = viewport_width*u;
    vec3 viewport_y = viewport_height*(-v);
    delta_x = viewport_x/image_width;
    delta_y = viewport_y/image_height;
    pixel00 = center - focal_length*w - viewport_x/2.f - viewport_y/2.f + 0.5f*(delta_x+delta_y);
}
__device__ color camera::ray_color(const ray& r, hittable_list** world, curandState* rand_state) const {
    ray curr_ray = r;
    color curr_attenuation = color(1.0f, 1.0f, 1.0f);
    for (int i=0; i<50; i++) {
        hit_record record;
        if ((*world)->hit(curr_ray, 0.001f, FLT_MAX, record)) {
            ray scattered;
            color attenuation;
            if (record.mat->scatter(curr_ray, record, attenuation, scattered, rand_state)) {
                curr_ray = scattered;
                curr_attenuation*=attenuation;
            } else return vec3{0,0,0};
        } else {
            vec3 dir = unit(curr_ray.dir());
            float a = 0.5f*(dir.y() + 1.0f);
            return curr_attenuation*((1.0f-a)*color(1.0f, 1.0f, 1.0f) + a*color(0.5f, 0.7f, 1.0f));
        }
    }
    return color{0.0f, 0.0f, 0.0f};
}