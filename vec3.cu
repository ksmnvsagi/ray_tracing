#include "vec3.cuh"
__device__ __host__ vec3::vec3():vec{0, 0, 0} {

}
__device__ __host__ vec3::vec3(float x, float y, float z):vec{x, y, z} {

}

__device__ __host__ float vec3::x() const {
    return vec[0];
}
__device__ __host__ float vec3::y() const {
    return vec[1];
}
__device__ __host__ float vec3::z() const {
    return vec[2];
}

__device__ __host__ vec3 vec3::operator-() const {
    return vec3{-vec[0], -vec[1], -vec[2]};
}
__device__ __host__ float vec3::operator[](int i) const {
    return vec[i];
}
__device__ __host__ float& vec3::operator[](int i) {
    return vec[i];
}
__device__ __host__ vec3& vec3::operator+=(const vec3& v) {
    vec[0]+=v.x();
    vec[1]+=v.y();
    vec[2]+=v.z();
    return *this;
}
__device__ __host__ vec3& vec3::operator-=(const vec3& v) {
    vec[0]-=v.x();
    vec[1]-=v.y();
    vec[2]-=v.z();
    return *this;
}
__device__ __host__ vec3& vec3::operator*=(const vec3& v) {
    vec[0]*=v.x();
    vec[1]*=v.y();
    vec[2]*=v.z();
    return *this;
}
__device__ __host__ vec3& vec3::operator*=(float t) {
    vec[0]*=t;
    vec[1]*=t;
    vec[2]*=t;
    return *this;
}
__device__ __host__ vec3& vec3::operator/=(float t) {
    return *this*=(float)(1/t);
}
__device__ __host__ float vec3::length() const {
    return sqrt(squared_length());
}
__device__ __host__ float vec3::squared_length() const {
    return vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2];
}

std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}
__device__ __host__ vec3 operator+(const vec3& v1, const vec3& v2) {
    return vec3{v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z()};
}
__device__ __host__ vec3 operator-(const vec3& v1, const vec3& v2) {
    return vec3{v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z()};
}
__device__ __host__ vec3 operator*(const vec3& v1, const vec3& v2) {
    return vec3{v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z()};
}
__device__ __host__ vec3 operator*(float t, const vec3& v) {
    return vec3{t * v.x(), t * v.y(), t * v.z()};
}
__device__ __host__ vec3 operator*(const vec3& v, float t) {
    return t * v;
}
__device__ __host__ vec3 operator/(const vec3& v, float t) {
    return (float)(1/t) * v;
}

__device__ __host__ float dot(const vec3& v1, const vec3& v2) {
    return v1.x() * v2.x()
           + v1.y() * v2.y()
           + v1.z() * v2.z();
}
__device__ __host__ vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3{v1.y() * v2.z() - v1.z() * v2.y(),
                v1.z() * v2.x() - v1.x() * v2.z(),
                v1.x() * v2.y() - v1.y() * v2.x()};
}
__device__ __host__ vec3 unit(const vec3& v) {
    return v / v.length();
}
__device__ __host__ bool near_zero(const vec3& v) {
    return (fabs(v.x()) < FLT_MIN) && (fabs(v.y()) < FLT_MIN) && (fabs(v.z()) < FLT_MIN);
}

__device__ vec3 randUnit(curandState* rand_state) {
    vec3 ret;
    do {
        ret = randVec(rand_state, -1, 1);
    } while (ret.squared_length() >= 1.0f);
    return unit(ret);
}
__device__ vec3 randVec(curandState* rand_state, float min, float max) {
    return vec3{cudaRand(rand_state, min, max), cudaRand(rand_state, min, max), cudaRand(rand_state, min, max)};
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f*dot(v,n)*n;
}
__device__ vec3 refract(const vec3& v, const vec3& n, float eta_ratio) {
    float cos_theta = fmin(dot(-v, n), 1.0f);
    vec3 refracted_perp = eta_ratio * (v + cos_theta*n);
    vec3 refracted_par = -sqrt(fabs(1.0f - refracted_perp.squared_length())) * n;
    return refracted_perp + refracted_par;
}


void write_color(std::ostream& out, const color& pixel_color) {
    int r = int(256 * clamp(GAMMA(pixel_color.x())));
    int g = int(256 * clamp(GAMMA(pixel_color.y())));
    int b = int(256 * clamp(GAMMA(pixel_color.z())));
    out << r << ' ' << g << ' ' << b << '\n';
}