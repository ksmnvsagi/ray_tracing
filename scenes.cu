#include "scenes.cuh"
#include "quad.cuh"
#include "instance.cuh"


__device__ void global(int size, hittable_list** world, bvh** node, curandState* rand_state) {
    *world = new hittable_list(size);
//        (*world)->add(new sphere(vec3(0.0f,-1000.0f, 0.0f), 1000.0f,
//                                 new lambertian(vec3(0.5f, 0.5f, 0.5f)))); // ground
    texture* checker = new checker_texture(0.32f, color{0.2f, 0.3f, 0.1f}, color{0.9f, 0.9f, 0.9f});
    (*world)->add(new sphere(vec3(0.0f,-1000.0f, 0.0f), 1000.0f,
                             new lambertian(checker))); // ground
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = curand_uniform(rand_state);
            vec3 center(a + 0.9f*curand_uniform(rand_state), 0.2f, b + 0.9f*curand_uniform(rand_state));
            if (choose_mat < 0.8f) {
                (*world)->add(new sphere(center, 0.2f,
                                         new lambertian(vec3(curand_uniform(rand_state)*curand_uniform(rand_state),
                                                             curand_uniform(rand_state)*curand_uniform(rand_state),
                                                             curand_uniform(rand_state)*curand_uniform(rand_state)))));
            } else if (choose_mat < 0.95f) {
                (*world)->add(new sphere(center, 0.2f,
                                         new metal(vec3(0.5f*(1.0f+curand_uniform(rand_state)),
                                                        0.5f*(1.0f+curand_uniform(rand_state)),
                                                        0.5f*(1.0f+curand_uniform(rand_state))),
                                                   curand_uniform(rand_state))));
            } else {
                (*world)->add(new sphere(center, 0.2f, new dielectric(1.5f)));
            }
        }
    }
    (*world)->add(new sphere(vec3(0, 1,0), 1.0f, new dielectric(1.5f)));
    (*world)->add(new sphere(vec3(-4, 1, 0), 1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f))));
    (*world)->add(new sphere(vec3(4, 1, 0), 1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f)));
    *node = new bvh(world, rand_state);
}

__device__ void checker_spheres(int size, hittable_list** world, bvh** node, curandState* rand_state) {
    *world = new hittable_list(size);
    texture* checker = new checker_texture(0.32f, color{0.2f, 0.3f, 0.1f}, color{0.9f, 0.9f, 0.9f});
    (*world)->add(new sphere(vec3(0,-10, 0), 10.0f, new lambertian(checker)));
    (*world)->add(new sphere(vec3(0,10, 0), 10.0f, new lambertian(checker)));
    *node = new bvh(world, rand_state);
}

__device__ void earth(int size, hittable_list** world, bvh** node, image* img, curandState* rand_state) {
    *world = new hittable_list(size);
    (*world)->add(new sphere(point3(0,0,0), 2, new lambertian(new image_texture(*img))));
    *node = new bvh(world, rand_state);
}

__device__ void quads(int size, hittable_list** world, bvh** node, curandState* rand_state) {
    *world = new hittable_list(size);
    (*world)->add(new quad(point3(-3,-2, 5), vec3(0, 0,-4), vec3(0, 4, 0), new lambertian(color(1, 0.2f, 0.2f))));
    (*world)->add(new quad(point3(-2,-2, 0), vec3(4, 0, 0), vec3(0, 4, 0), new lambertian(color(0.2, 1.0, 0.2)))); // back
    (*world)->add(new quad(point3( 3,-2, 1), vec3(0, 0, 4), vec3(0, 4, 0), new lambertian(color(0.2, 0.2, 1.0))));
    (*world)->add(new quad(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), new lambertian(color(1.0, 0.5, 0.0))));
    (*world)->add(new quad(point3(-2,-3, 5), vec3(4, 0, 0), vec3(0, 0,-4), new lambertian(color(0.2, 0.8, 0.8))));
    *node = new bvh(world, rand_state);
}

__device__ void empty_cornell(int size, hittable_list** world, bvh** node, image* img, curandState* rand_state) {
    *world = new hittable_list(size);
    (*world)->add(new quad(point3(555,0,0), vec3(0,555,0), vec3(0,0,555), new lambertian(color(0.12f, 0.45f, 0.15f)))); // green
    (*world)->add(new quad(point3(0,0,0), vec3(0,555,0), vec3(0,0,555), new lambertian(color(0.65f, 0.05f, 0.05f)))); // red
    (*world)->add(new quad(point3(343, 554, 332), vec3(-130,0,0), vec3(0,0,-105), new diffuse_light(color(15.f, 15.f, 15.f)))); // light
    (*world)->add(new quad(point3(0,0,0), vec3(555,0,0), vec3(0,0,555), new lambertian(color(0.73f, 0.73f, 0.73f)))); // bottom
    (*world)->add(new quad(point3(555,555,555), vec3(-555,0,0), vec3(0,0,-555), new lambertian(color(0.73f, 0.73f, 0.73f)))); // top
    (*world)->add(new quad(point3(0,0,555), vec3(555,0,0), vec3(0,555,0), new lambertian(color(0.73f, 0.73f, 0.73f)))); // center

    (*world)->add(new sphere(point3(190,90,190), 90, new dielectric(1.5f)));
//    (*world)->add(new sphere(point3(190,90,190), 90, new lambertian(new image_texture(*img))));
    hittable* box1 = create_box(point3(0,0,0), point3(165,330,165), new lambertian(color(0.73f, 0.73f, 0.73f)));
    box1 = new rotate_y(box1, 15);
    box1 = new translate(box1, vec3(265,0,295));
//    hittable* box2 = create_box(point3(0,0,0), point3(165,165,165), new lambertian(color(0.73f, 0.73f, 0.73f)));
//    box2 = new rotate_y(box2, -18);
//    box2 = new translate(box2, vec3(130,0,65));
    (*world)->add(box1);
//    (*world)->add(box2);

    *node = new bvh(world, rand_state);
}