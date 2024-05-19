#include "config.cuh"
#include "camera.cuh"
#include "scenes.cuh"
#include "quad.cuh"

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

__global__ void create_world(int size, hittable_list** world, bvh** node, image* img, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        global(size, world, node, rand_state);
//        checker_spheres(size, world, node, rand_state);
//        earth(size, world, node, img, rand_state);
//        quads(size, world, node, rand_state);
        empty_cornell(size, world, node, img, rand_state);
    }
}

__global__ void free_world(int size, hittable_list** world, bvh** node) {
    for(int i=0; i < size; i++) {
//        delete ((quad*)list[i])->mat;
//        delete list[i];
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
    // to handle recursion during BVH construction
    size_t stack_size = 8192;
    cudaDeviceSetLimit(cudaLimitStackSize, stack_size);

    const unsigned int WIDTH = 600;
    const unsigned int HEIGHT = 600;
    const unsigned int WORLD_SIZE = 8;
    // query device
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int SMs = prop.multiProcessorCount;
    std::cout<<"SMs: "<<prop.multiProcessorCount<<"/nCompute capability: "<<prop.major<<'.'<<prop.minor<<'\n';

    int num_pixels = WIDTH*HEIGHT;
    // host and device color buffers
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

    // world creation (must be done on the GPU due to virtual functions!)
    hittable_list** world;
    cudaCheck(cudaMalloc((void**)&world, sizeof(hittable*)));
    bvh** node;
    cudaCheck(cudaMalloc((void**)&node, sizeof(bvh*)));
    // camera
    camera cam(1.0f, WIDTH, point3(278, 278, -800), point3(278, 278, 0), 40, 100);
    image host_earth_texture("..\\earthmap.jpg");
    image* dev_earth_texture;
    cudaCheck(cudaMalloc((void**)&dev_earth_texture, sizeof(image)));
    cudaCheck(cudaMemcpy(dev_earth_texture, &host_earth_texture, sizeof(image), cudaMemcpyHostToDevice));
    create_world<<<1,1>>>(WORLD_SIZE, world, node, dev_earth_texture, rand_states);

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
    free_world<<<1,1>>>(WORLD_SIZE, world, node);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFree(dev_buff));
    cudaCheck(cudaFree(rand_states));
    cudaCheck(cudaFree(world));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
}