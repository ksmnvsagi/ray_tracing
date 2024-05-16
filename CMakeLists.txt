cmake_minimum_required(VERSION 3.17)
project(ray_tracing CUDA C CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ray_tracing
        main.cu
        config.cuh
        vec3.cu
        vec3.cuh
        ray.cu
        ray.cuh
        hittable.cuh
        sphere.cu
        sphere.cuh
        hittable_list.cu
        hittable_list.cuh
        material.cu
        material.cuh
        camera.cu
        camera.cuh
        aabb.cu
        aabb.cuh
        bvh.cu
        bvh.cu
)
set_target_properties(ray_tracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(ray_tracing PROPERTIES CUDA_ARCHITECTURES "86")