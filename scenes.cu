#include "scenes.cuh"


__device__ void global(int size, hittable** list, hittable_list** world, bvh** node, curandState* rand_state) {
    *world = new hittable_list(list, size);
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

__device__ void checker_spheres(int size, hittable** list, hittable_list** world, bvh** node, curandState* rand_state) {
    *world = new hittable_list(list, size);
    texture* checker = new checker_texture(0.32f, color{0.2f, 0.3f, 0.1f}, color{0.9f, 0.9f, 0.9f});
    (*world)->add(new sphere(vec3(0,-10, 0), 10.0f, new lambertian(checker)));
    (*world)->add(new sphere(vec3(0,10, 0), 10.0f, new lambertian(checker)));
    *node = new bvh(world, rand_state);
}

__device__ void earth(int size, hittable** list, hittable_list** world, bvh** node, image* img, curandState* rand_state) {
    *world = new hittable_list(list, size);
    (*world)->add(new sphere(point3(0,0,0), 2, new lambertian(new image_texture(*img))));
    *node = new bvh(world, rand_state);
}