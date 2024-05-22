#include "image.cuh"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.cuh"


__host__ image::image() {

}
__host__ image::image(const char* file) {
    int n = bytes_per_pixel;
    fdata = stbi_loadf(file, &image_width, &image_height, &n, bytes_per_pixel);
    bytes_per_line = image_width * bytes_per_pixel;

    int total_bytes = image_width * image_height * bytes_per_pixel;
    host_bdata = new unsigned char[total_bytes];

    unsigned char* bptr = host_bdata;
    float* fptr = fdata;
    for (int i=0; i < total_bytes; i++, fptr++, bptr++) *bptr = to_byte(*fptr);
    cudaCheck(cudaMalloc((void**)&dev_bdata, total_bytes));
    cudaCheck(cudaMemcpy(dev_bdata, host_bdata, total_bytes, cudaMemcpyHostToDevice));
    delete[] host_bdata;
}
__device__ unsigned char* image::pixel(int x, int y) const {
    x = pixel_clamp(x, 0, image_width);
    y = pixel_clamp(y, 0, image_height);
    return dev_bdata + y*bytes_per_line + x*bytes_per_pixel;
}
__device__ int image::pixel_clamp(int val, int min, int max) const {
    if (val < min) return min;
    if (val < max) return val;
    return max - 1;
}
__host__ unsigned char image::to_byte(float val) const {
    if (val <= 0.0f) return 0;
    if (1.0f <= val) return 255;
    return static_cast<unsigned char>(256.0f * val);
}