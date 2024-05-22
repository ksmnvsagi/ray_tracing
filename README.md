# CUDA-accelerated Ray Tracer

This project is a CUDA-accelerated ray tracer, inspired by the books 
[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) 
and [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
by Peter Shirley. 
The implementation leverages the parallel processing power of an NVIDIA GPU to achieve efficient rendering 
of 3D scenes. Testing and debugging were done on NVIDIA 3060 Ti GPU (38 Streaming multiprocessors)
using CUDA 8.6. Benchmarking showed speedup of 24 times compared to CPU-only implementation of the 
same ray tracer.


### Cornell Box with a glass ball (1000 rays per pixel, 600x600):
![image](https://github.com/ksmnvsagi/ray_tracing/assets/169926151/16dbd077-1b1d-475f-85d8-3d2fcc55c2bc)

### Diffuse (Lambertian), dielectric and metal materials (100 rays per pixel, 1200x800)
![image-_1_](https://github.com/ksmnvsagi/ray_tracing/assets/169926151/3ec01a65-2c2a-4e22-b7bc-685253603b1d)

### TODO:
- Parallelize the application by introducing multiple CUDA streams
- Exploit shared memory and constant memory in CUDA to achieve greater speed up
- Implement support for object loading (triangles already supported!)
