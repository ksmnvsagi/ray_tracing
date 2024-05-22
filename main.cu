#include "config.cuh"
#include "camera.cuh"
#include "scenes.cuh"
#include "quad.cuh"
#include "OBJ_Loader.h"


__global__ void render(camera cam, color* buff, bvh** node, curandState* rand_states) {
    for (int y = threadIdx.y + blockIdx.y * blockDim.y; y<cam.image_height; y+=blockDim.y*gridDim.y) {
        for (int x = threadIdx.x + blockIdx.x * blockDim.x; x<cam.image_width; x+=blockDim.x*gridDim.x) {
            int index = x + y*cam.image_width;
            curandState my_state = rand_states[index];
            point3 pixel = cam.pixel00 + x*cam.delta_x + y*cam.delta_y;
            color final_color(0, 0, 0);
            for (int i=0; i<cam.num_samples; i++) {
                vec3 ray_dir = pixel + 0.5f*cam.delta_x * cudaRand(&my_state, -1, 1) +
                               0.5f*cam.delta_y*cudaRand(&my_state, -1, 1) - cam.center;
                ray r(cam.center,ray_dir);
                final_color+=cam.ray_color(r, *node, &my_state);
            }
            buff[index] = final_color/(float)cam.num_samples;
        }
    }
}

__global__ void create_world(int size, bvh** node, vec3* vertices, vec3* triangles, int total_triangles, image* img, curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
//        cornell(size, node, img, rand_state);
        obj_test(size, node, vertices, triangles, total_triangles);
    }
}

__global__ void free_world(int size, bvh** node) {
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
    objl::Loader Loader;
    bool loadout = Loader.LoadFile("../cow.obj");
    objl::Mesh currMesh = Loader.LoadedMeshes[0];
    int total_vertices = currMesh.Vertices.size();
    int total_triangles = currMesh.Indices.size()/3;
    vec3* vertices = (vec3*)malloc(total_vertices*sizeof(vec3));
    vec3* indices = (vec3*)malloc(total_triangles*sizeof(vec3));
    for (int j = 0; j < currMesh.Vertices.size(); j++) {
//        std::cout<<"V("<<currMesh.Vertices[j].Position.X<<", "<<currMesh.Vertices[j].Position.Y<<", "<<currMesh.Vertices[j].Position.Z<<")\n";
        vertices[j] = vec3(currMesh.Vertices[j].Position.X, currMesh.Vertices[j].Position.Y, currMesh.Vertices[j].Position.Z);
    }
    for (int j = 0; j < currMesh.Indices.size(); j += 3) {
//        std::cout<<"I("<<currMesh.Indices[j]<<", "<<currMesh.Indices[j+1]<<", "<<currMesh.Indices[j+2]<<")\n";
        indices[j/3] = vec3(currMesh.Indices[j], currMesh.Indices[j + 1], currMesh.Indices[j + 2]);
    }
    vec3* dev_vertices;
    cudaCheck(cudaMalloc((void**)&dev_vertices, total_vertices*sizeof(vec3)));
    cudaCheck(cudaMemcpy(dev_vertices, vertices, total_vertices*sizeof(vec3), cudaMemcpyHostToDevice));
    vec3* dev_triangles;
    cudaCheck(cudaMalloc((void**)&dev_triangles, total_triangles*sizeof(vec3)));
    cudaCheck(cudaMemcpy(dev_triangles, indices, total_triangles*sizeof(vec3), cudaMemcpyHostToDevice));

























    // to handle recursion during BVH construction
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    // initialization
    const unsigned int WIDTH = 600;
    const unsigned int HEIGHT = 600;
    const unsigned int WORLD_SIZE = 7;
    const float ASPECT_RATIO = 1.0f;
    const point3 lookat = point3(2500/2, 2500/2, 0);
    const point3 lookfrom = point3(2500/2, 2500/2, -4500);
//    const point3 lookat = point3(0, 0, 0);
//    const point3 lookfrom = point3(4000, 4000, 4000);
    const float vfov = 40.0f;
    const unsigned int SAMPLES = 1000;
    const int NUM_PIXELS = WIDTH*HEIGHT;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    camera cam(ASPECT_RATIO, WIDTH, lookfrom, lookat, vfov, SAMPLES);

    // query device
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int SMs = prop.multiProcessorCount;
    std::cout<<"SMs: "<<prop.multiProcessorCount<<"\nCompute capability: "<<prop.major<<'.'<<prop.minor<<'\n';

    // host and device color buffers
    color* host_buff = (color*)malloc(NUM_PIXELS*sizeof(color));
    color* dev_buff;
    cudaCheck(cudaMalloc((void**)&dev_buff, NUM_PIXELS*sizeof(color)));

    dim3 blocks(2*SMs, 2*SMs);
    dim3 threads(32, 32);
//    dim3 blocks(1, 1);
//    dim3 threads(1, 1);

    // curand init
    curandState* rand_states;
    cudaCheck(cudaMalloc((void**)&rand_states, NUM_PIXELS*sizeof(curandState)));
    create_rand<<<blocks, threads>>>(WIDTH, HEIGHT, rand_states);

    // world creation (must be done on the GPU due to virtual functions)
    bvh** node;
    cudaCheck(cudaMalloc((void**)&node, sizeof(bvh*)));
    image host_texture("..\\earthmap.jpg");
    image* dev_texture;
    cudaCheck(cudaMalloc((void**)&dev_texture, sizeof(image)));
    cudaCheck(cudaMemcpy(dev_texture, &host_texture, sizeof(image), cudaMemcpyHostToDevice));
    create_world<<<1,1>>>(WORLD_SIZE, node, dev_vertices, dev_triangles, total_triangles, dev_texture, rand_states);
    cudaCheck(cudaDeviceSynchronize());
    // render
    cudaCheck(cudaEventRecord(start));
    render<<<blocks, threads>>>(cam, dev_buff, node, rand_states);
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed;
    cudaCheck(cudaEventElapsedTime(&elapsed, start, stop));
    std::cout<<"Elapsed time: "<<elapsed<<" ms\n";
    // copy memory back to CPU
    cudaCheck(cudaMemcpy(host_buff, dev_buff, NUM_PIXELS*sizeof(color), cudaMemcpyDeviceToHost));

    // output image
    std::ofstream output("../image.ppm");
    output << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    for (int i=0; i<NUM_PIXELS; i++) write_color(output, host_buff[i]);
    output.close();

    // cleanup
    free_world<<<1,1>>>(WORLD_SIZE, node);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaFree(dev_buff));
    cudaCheck(cudaFree(rand_states));
    cudaCheck(cudaFree(node));
    cudaCheck(cudaFree(dev_texture));
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
}