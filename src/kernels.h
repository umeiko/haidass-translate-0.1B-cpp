// kernels.h — vector dot kernels (weight dtype x f32 activations) + dispatch.
#pragma once

#include <cstdint>
#include <cstring>

namespace haidass {

// f16 (IEEE half) -> f32
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        // subnormal: value = mant * 2^-24
        float val = (float)mant * 5.9604644775390625e-08f;
        uint32_t bits;
        std::memcpy(&bits, &val, 4);
        f = sign | bits;
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// bf16 -> f32 (pure bit shift)
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// f32 -> f16 (round to nearest even, enough for quant scales)
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x & 0x80000000u) >> 16;
    const int exp = (int)((x >> 23) & 0xFF);
    uint32_t mant = x & 0x7FFFFFu;
    if (exp == 255) return (uint16_t)(sign | 0x7C00u | (mant ? 1 : 0));  // inf/nan
    int e = exp - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);           // overflow -> inf
    if (e <= 0) {                                              // subnormal/underflow
        if (e < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t m = mant >> (unsigned)(14 - e);
        if ((mant >> (unsigned)(13 - e)) & 1) m += 1;
        return (uint16_t)(sign | m);
    }
    uint32_t m = mant >> 13;
    if (mant & 0x1000u) m += 1;  // round to nearest
    return (uint16_t)(sign | ((uint32_t)e << 10) | (m & 0x3FFu));
}

// Dot products: y (weights, encoded) . x (f32 activations), n elements.
// For quantized dtypes n must be a multiple of the block size (32).
struct Kernels {
    float (*dot_f32)(const float* y, const float* x, int64_t n);
    float (*dot_f16)(const uint16_t* y, const float* x, int64_t n);
    float (*dot_bf16)(const uint16_t* y, const float* x, int64_t n);
    float (*dot_q8_0)(const uint8_t* y, const float* x, int64_t n);
    float (*dot_q4_0)(const uint8_t* y, const float* x, int64_t n);
};

// Active kernel table (filled by kernels_init(), defaults to scalar).
extern Kernels g_kernels;

// Detect CPU features once and install the best available kernels.
void kernels_init();

// Name of the active SIMD tier ("scalar", "avx2", "neon", ...), for diagnostics.
const char* kernels_active_tier();

// Implementation entry points (compiled in separate TUs with ISA flags).
void kernels_scalar_fill(Kernels* k);
#if defined(HAIDASS_HAVE_AVX2)
void kernels_avx2_fill(Kernels* k);
#endif
#if defined(HAIDASS_HAVE_NEON)
void kernels_neon_fill(Kernels* k);
#endif

} // namespace haidass
