#include "scenes.cuh"
#include "quad.cuh"
#include "instance.cuh"

__device__ void cornell(int size, bvh** node) {
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
    *node = new bvh(world);
}
__device__ void tri_test(int size, bvh** node) {
    hittable_list* world = new hittable_list(size);
    (world)->add(new quad(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), new lambertian(color(0.12f, 0.45f, 0.15f)))); // green
    (world)->add(new quad(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), new lambertian(color(0.65f, 0.05f, 0.05f)))); // red
    (world)->add(new quad(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), new diffuse_light(color(15.f, 15.f, 15.f)))); // light
    (world)->add(new quad(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), new lambertian(color(0.73f, 0.73f, 0.73f)))); // bottom
    (world)->add(new quad(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), new lambertian(color(0.73f, 0.73f, 0.73f)))); // top
    (world)->add(new quad(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), new lambertian(color(0.73f, 0.73f, 0.73f)))); // center
    (world)->add(new tri(point3(190,90,190), vec3(200,0,0), vec3(0,200,0), new lambertian(color(0.73f, 0.73f, 0.73f))));
    hittable* box1 = create_box(point3(0,0,0), point3(165,330,165), new lambertian(color(0.73f, 0.73f, 0.73f)));
    box1 = new rotate_y(box1, 15);
    box1 = new translate(box1, vec3(265,0,295));
    (world)->add(box1);
    *node = new bvh(world);
}
__device__ void obj_test(int size, bvh** node, vec3* vertices, vec3* triangles, int total_triangles) {
    hittable_list* world = new hittable_list(size);
    (world)->add(new quad(point3(2500,0,0), vec3(0,2500,0), vec3(0,0,2500), new lambertian(color(0.12f, 0.45f, 0.15f)))); // green
    (world)->add(new quad(point3(0,0,0), vec3(0,2500,0), vec3(0,0,2500), new lambertian(color(0.65f, 0.05f, 0.05f)))); // red
    (world)->add(new quad(point3(2500/2-500/2, 2499, 2500/2), vec3(500,0,0), vec3(0,0,-500), new diffuse_light(color(15.f, 15.f, 15.f)))); // light
    (world)->add(new quad(point3(0,0,0), vec3(2500,0,0), vec3(0,0,2500), new lambertian(color(0.73f, 0.73f, 0.73f)))); // bottom
    (world)->add(new quad(point3(2500,2500,2500), vec3(-2500,0,0), vec3(0,0,-2500), new lambertian(color(0.73f, 0.73f, 0.73f)))); // top
    (world)->add(new quad(point3(0,0,2500), vec3(2500,0,0), vec3(0,2500,0), new lambertian(color(0.73f, 0.73f, 0.73f)))); // center

//    hittable* obj = create_object(vertices, triangles, total_triangles, new metal(color(1.f, 0.f, 0.f), 0.2f));
    hittable* obj = create_object(vertices, triangles, total_triangles, new lambertian(color(0.f, 0.f, 1.f)));

    obj = new translate(obj, vec3(2500/2-1500 ,0,2500/2+500));
    obj = new rotate_y(obj, 45);
    (world)->add(obj);
    *node = new bvh(world);
}