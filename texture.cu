#include "texture.cuh"

__device__ __host__ solid_color::solid_color(const color& albedo):albedo(albedo) {

}

__device__ __host__ solid_color::solid_color(float r, float g, float b):solid_color(color{r, g, b}) {

}

__device__ __host__ color solid_color::value(float u, float v, const point3& p) const {
    return albedo;
}

__device__ __host__ checker_texture::checker_texture(float scale, texture* even, texture* odd): inv_scale(1.f / scale),
even(even), odd(odd) {

}
__device__ __host__ checker_texture::checker_texture(float scale, const color& c1, const color& c2): inv_scale(1.f/scale),
even(new solid_color(c1)), odd(new solid_color(c2)) {


}
__device__ __host__ color checker_texture::value(float u, float v, const point3& p) const {
    int xInteger = int(floor(inv_scale * p.x()));
    int yInteger = int(floor(inv_scale * p.y()));
    int zInteger = int(floor(inv_scale * p.z()));
    bool isEven = (xInteger + yInteger + zInteger)%2 == 0;
    return isEven ? even->value(u, v, p) : odd->value(u, v, p);
}