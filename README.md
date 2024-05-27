# CUDA-accelerated Ray Tracer

This project is a CUDA-accelerated ray tracer, inspired by the books 
[Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) 
and [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html)
by Peter Shirley. 
The implementation leverages the parallel processing power of an NVIDIA GPU to achieve efficient rendering 
of 3D scenes. Testing and debugging were done on NVIDIA 3060 Ti GPU (38 Streaming multiprocessors)
using CUDA 8.6. Benchmarking showed speedup of 24 times compared to CPU-only implementation of the 
same ray tracer. External libraries used are stb_image and OBJ_Loaded to facilitate texture handling and objects loading. 


### Cornell Box with a glass ball (1000 rays per pixel, 50 bounces, 600x600 image):
![image](https://github.com/ksmnvsagi/ray_tracing/assets/169926151/16dbd077-1b1d-475f-85d8-3d2fcc55c2bc)

### Cornell Box with a model of a cow (720 primitives, 1000 rays per pixel, 50 bounces, 600x600 image):
![image (1)](https://github.com/ksmnvsagi/ray_tracing/assets/169926151/c51e809a-f625-40e9-9763-b248339dd1e1)

### Diffuse (Lambertian), dielectric and metal materials (430 primitives, 100 rays per pixel, 50 bounces, 1200x800 image)
![image-_1_](https://github.com/ksmnvsagi/ray_tracing/assets/169926151/3ec01a65-2c2a-4e22-b7bc-685253603b1d)

### TODO:
- Exploit shared memory and constant memory in CUDA to achieve greater speed up
- Implement Monte Carlo sampling technique to reduce the required number of samples per pixel
