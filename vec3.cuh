#pragma once
#include "config.cuh"

class vec3 {
public:
    __device__ __host__ vec3();
    __device__ __host__ vec3(float x, float y, float z);

    __device__ __host__ float x() const;
    __device__ __host__ float y() const;
    __device__ __host__ float z() const;

    __device__ __host__ vec3 operator-() const;
    __device__ __host__ float operator[](int i) const;
    __device__ __host__ float& operator[](int i);
    __device__ __host__ vec3& operator+=(const vec3& v);
    __device__ __host__ vec3& operator-=(const vec3& v);
    __device__ __host__ vec3& operator*=(const vec3& v);
    __device__ __host__ vec3& operator*=(float t);
    __device__ __host__ vec3& operator/=(float t);
    __device__ __host__ float length() const;
    __device__ __host__ float squared_length() const;
private:
    float vec[3];
};

using point3 = vec3;

std::ostream& operator<<(std::ostream& out, const vec3& v);
__device__ __host__ vec3 operator+(const vec3& v1, const vec3& v2);
__device__ __host__ vec3 operator-(const vec3& v1, const vec3& v2);
__device__ __host__ vec3 operator*(const vec3& v1, const vec3& v2);
__device__ __host__ vec3 operator*(float t, const vec3& v);
__device__ __host__ vec3 operator*(const vec3& v, float t);
__device__ __host__ vec3 operator/(const vec3& v, float t);

__device__ __host__ float dot(const vec3& v1, const vec3& v2);
__device__ __host__ vec3 cross(const vec3& v1, const vec3& v2);
__device__ __host__ vec3 unit(const vec3& v);
__device__ __host__ bool near_zero(const vec3& v);

__device__ vec3 randUnit(curandState* rand_state);
__device__ vec3 randVec(curandState* rand_state, float min = 0.f, float max = 1.f);

__device__ vec3 reflect(const vec3& v, const vec3& n);
__device__ vec3 refract(const vec3& v, const vec3& n, float eta_ratio);

using color = vec3;

void write_color(std::ostream& out, const color& pixel_color);
