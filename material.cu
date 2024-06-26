#include "material.cuh"
__device__ lambertian::lambertian(const color& albedo):albedo(albedo) {

}
__device__ lambertian::lambertian(texture* tex):tex(tex) {

}
__device__ bool lambertian::scatter(const ray& r, const hit_record& record, color& attenuation,
                                    ray& scattered, curandState* rand_state) const {
    vec3 scattered_dir = record.normal+randUnit(rand_state);
    if (near_zero(scattered_dir)) scattered_dir = record.normal;
    scattered = ray(record.p, scattered_dir);
    attenuation = tex ? tex->value(record.u, record.v, record.p) : albedo;
    return true;
}
__device__ color lambertian::emit(float u, float v, const point3& p) const {
    return color{0, 0, 0};
}

__device__ metal::metal(const color& albedo, float fuzz):albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {

}
__device__ bool metal::scatter(const ray& r, const hit_record& record, color& attenuation,
                               ray& scattered, curandState* rand_state) const {
    vec3 reflected = reflect(r.dir(), record.normal);
    reflected = reflected + fuzz*randUnit(rand_state);
    scattered = ray(record.p, reflected);
    attenuation = albedo;
    return true;
}
__device__ color metal::emit(float u, float v, const point3& p) const {
    return color{0, 0, 0};
}

__device__ dielectric::dielectric(float refraction_index):refraction_index(refraction_index) {

}
__device__ bool dielectric::scatter(const ray& r, const hit_record& record, color& attenuation,
                                    ray& scattered, curandState* rand_state) const {
    attenuation = color(1.0f, 1.0f, 1.0f); // dielectrics absorb no colors
    vec3 outward_normal = record.front_face ? record.normal : -record.normal;
    float ri = record.front_face ? (1.0f/refraction_index) : refraction_index;
    float cos_theta = fmin(dot(-unit(r.dir()), outward_normal), 1.0f); // angle between incoming ray and normal
    float sin_theta = sqrt(1.0f - cos_theta*cos_theta);
    vec3 scattered_dir;
    if (ri*sin_theta > 1.0f || schlick(cos_theta, ri) > cudaRand(rand_state)) // if cannot refract
        scattered_dir = reflect(unit(r.dir()), outward_normal);
    else scattered_dir = refract(unit(r.dir()), outward_normal, ri);
    scattered = ray(record.p, scattered_dir);
    return true;
}
__device__ color dielectric::emit(float u, float v, const point3& p) const {
    return color{0, 0, 0};
}
__device__ float dielectric::schlick(float cosine, float ri) const {
    float r0 = (1 - ri)/(1 + ri);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine), 5);
}

__device__ diffuse_light::diffuse_light(const color& emitted_color):tex(new solid_color(emitted_color)) {

}
__device__ diffuse_light::diffuse_light(texture* tex):tex(tex) {

}
__device__ color diffuse_light::emit(float u, float v, const point3& p) const {
    return tex->value(u, v, p);
}
__device__ bool diffuse_light::scatter(const ray& r, const hit_record& record, color& attenuation,
                        ray& scattered, curandState* rand_state) const {
    return false;
}