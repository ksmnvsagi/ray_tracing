#pragma once
#include "config.cuh"

class image {
public:
    __host__ image();
    __host__ image(const char* file);
    __device__ unsigned char* pixel(int x, int y) const;
    __device__ int pixel_clamp(int val, int min, int max) const;
    __host__ const unsigned char to_byte(float val) const;

    int image_width = 0;
    int image_height = 0;
private:
    const int bytes_per_pixel = 3;
    float* fdata;
    unsigned char* host_bdata;
    unsigned char* dev_bdata;
    int bytes_per_line = 0;
};