#include "scenes.cuh"
#include "quad.cuh"
#include "instance.cuh"

__device__ void cornell(int size, bvh** node, image* img, curandState* rand_state) {
    hittable_list* world = new hittable_list(size);
    (world)->add(new quad(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), new lambertian(color(0.12f, 0.45f, 0.15f)))); // green
    (world)->add(new quad(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), new lambertian(color(0.65f, 0.05f, 0.05f)))); // red
    (world)->add(new quad(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), new diffuse_light(color(15.f, 15.f, 15.f)))); // light
    (world)->add(new quad(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), new lambertian(color(0.73f, 0.73f, 0.73f)))); // bottom
    (world)->add(new quad(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), new lambertian(color(0.73f, 0.73f, 0.73f)))); // top
    (world)->add(new quad(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), new lambertian(color(0.73f, 0.73f, 0.73f)))); // center
    (world)->add(new sphere(point3(190,90,190), 90, new dielectric(1.5f)));
    hittable* box1 = create_box(point3(0,0,0), point3(165,330,165), new lambertian(color(0.73f, 0.73f, 0.73f)));
    box1 = new rotate_y(box1, 15);
    box1 = new translate(box1, vec3(265,0,295));
    (world)->add(box1);
    *node = new bvh(world, rand_state);
}