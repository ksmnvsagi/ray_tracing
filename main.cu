#include "config.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "bvh.cuh"

__global__ void render(camera cam, color* buff, bvh** node, curandState* rand_states) {
    for (int y = threadIdx.y + blockIdx.y * blockDim.y; y<cam.image_height; y+=blockDim.y*gridDim.y) {
        for (int x = threadIdx.x + blockIdx.x * blockDim.x; x<cam.image_width; x+=blockDim.x*gridDim.x) {
            int index = x + y*cam.image_width;
            curandState my_state = rand_states[index];
            point3 pixel = (cam).pixel00 + x*(cam).delta_x + y*(cam).delta_y;
            color final_color(0, 0, 0);
            for (int i=0; i<(cam).num_samples; i++) {
                vec3 ray_dir = pixel + 0.5f*(cam).delta_x * cudaRand(&my_state, -1, 1) +
                               0.5f*(cam).delta_y*cudaRand(&my_state, -1, 1) - (cam).center;
                ray r((cam).center,ray_dir);
                final_color+=(cam).ray_color(r, *node, &my_state);
            }
            buff[index] = final_color/(float)(cam).num_samples;
        }
    }
}

__global__ void create_world(hittable** list, hittable_list** world, bvh** node, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *world = new hittable_list(list, 22*22+1+3);
        (*world)->add(new sphere(vec3(0.0f,-1000.0f,-1.0f), 1000.0f,
                                 new lambertian(vec3(0.5f, 0.5f, 0.5f)))); // ground
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = curand_uniform(rand_state);
                vec3 center(a+curand_uniform(rand_state),0.2f, b+curand_uniform(rand_state));
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
                                                         0.5f*curand_uniform(rand_state))));
                } else {
                    (*world)->add(new sphere(center, 0.2f, new dielectric(1.5f)));
                }
            }
        }
        (*world)->add(new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5)));
        (*world)->add(new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1))));
        (*world)->add(new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0)));
        *node = new bvh(world, rand_state);
    }
}

__global__ void free_world(hittable** list, hittable_list** world, bvh** node) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere*)list[i])->mat;
        delete list[i];
    }
    delete *world;
    delete *node;
}

__global__ void create_rand(int WIDTH, int HEIGHT, curandState* rand_states) {
    for (int y = threadIdx.y + blockIdx.y * blockDim.y; y<HEIGHT; y+=blockDim.y*gridDim.y) {
        for (int x = threadIdx.x + blockIdx.x * blockDim.x; x<WIDTH; x+=blockDim.x*gridDim.x) {
            int index = x + y*WIDTH;
            curand_init(1984+index, 0, 0, &rand_states[index]);
        }
    }
}

int main() {
    size_t stack_size = 8192;
    cudaDeviceSetLimit(cudaLimitStackSize, stack_size);

    const int WIDTH = 1200;
    const int HEIGHT = 800;
    // query device
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int SMs = prop.multiProcessorCount;

    int num_pixels = WIDTH*HEIGHT;
    color* host_buff = (color*)malloc(num_pixels*sizeof(color));
    color* dev_buff;
    cudaCheck(cudaMalloc((void**)&dev_buff, num_pixels*sizeof(color)));
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    dim3 blocks(2*SMs, 2*SMs);
    dim3 threads(32, 32);

    // curand init
    curandState* rand_states;
    cudaCheck(cudaMalloc((void**)&rand_states, num_pixels*sizeof(curandState)));
    create_rand<<<blocks, threads>>>(WIDTH, HEIGHT, rand_states);

    // world creation (must be done on the GPU due to virtual functions)
    hittable** list;
    cudaCheck(cudaMalloc((void**)&list, (22*22+1+3)*sizeof(hittable*)));
    hittable_list** world;
    cudaCheck(cudaMalloc((void**)&world, sizeof(hittable*)));
    bvh** node;
    cudaCheck(cudaMalloc((void**)&node, sizeof(bvh*)));
    camera cam(1.5f, 1200, point3{13, 2, 3}, point3{0, 0, 0}, 30, 10);
    create_world<<<1,1>>>(list, world, node, rand_states);

    // render
    cudaCheck(cudaEventRecord(start));
    render<<<blocks, threads>>>(cam, dev_buff, node, rand_states);
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed;
    cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));
    std::cout<<"Elapsed time: "<<elapsed<<" ms\n";

    // copy memory back to CPU
    cudaCheck(cudaMemcpy(host_buff, dev_buff, num_pixels*sizeof(color), cudaMemcpyDeviceToHost));

    // output image
    std::ofstream output("../image.ppm");
    output << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    for (int i=0; i<num_pixels; i++) write_color(output, host_buff[i]);
    output.close();
    // cleanup
    free_world<<<1,1>>>(list, world, node);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFree(dev_buff));
    cudaCheck(cudaFree(rand_states));
    cudaCheck(cudaFree(list));
    cudaCheck(cudaFree(world));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
}