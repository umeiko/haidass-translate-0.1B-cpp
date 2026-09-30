// kernels_scalar.cpp — portable reference dot kernels. Always available.
#include "kernels.h"

#include <cstddef>
#include <cstring>

namespace haidass {
namespace {

float dot_f32(const float* y, const float* x, int64_t n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += y[i + 0] * x[i + 0];
        s1 += y[i + 1] * x[i + 1];
        s2 += y[i + 2] * x[i + 2];
        s3 += y[i + 3] * x[i + 3];
    }
    for (; i < n; ++i) s0 += y[i] * x[i];
    return (s0 + s1) + (s2 + s3);
}

float dot_f16(const uint16_t* y, const float* x, int64_t n) {
    float s0 = 0, s1 = 0;
    int64_t i = 0;
    for (; i + 2 <= n; i += 2) {
        s0 += f16_to_f32(y[i + 0]) * x[i + 0];
        s1 += f16_to_f32(y[i + 1]) * x[i + 1];
    }
    for (; i < n; ++i) s0 += f16_to_f32(y[i]) * x[i];
    return s0 + s1;
}

float dot_bf16(const uint16_t* y, const float* x, int64_t n) {
    float s0 = 0, s1 = 0;
    int64_t i = 0;
    for (; i + 2 <= n; i += 2) {
        s0 += bf16_to_f32(y[i + 0]) * x[i + 0];
        s1 += bf16_to_f32(y[i + 1]) * x[i + 1];
    }
    for (; i < n; ++i) s0 += bf16_to_f32(y[i]) * x[i];
    return s0 + s1;
}

// Q8_0 block: { f16 scale; int8 q[32] }
float dot_q8_0(const uint8_t* y, const float* x, int64_t n) {
    float sum = 0.0f;
    const int64_t nblocks = n / 32;
    for (int64_t b = 0; b < nblocks; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, y, 2);
        const float d = f16_to_f32(d_bits);
        const int8_t* q = (const int8_t*)(y + 2);
        const float* xb = x + b * 32;
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int j = 0; j < 32; j += 4) {
            s0 += (float)q[j + 0] * xb[j + 0];
            s1 += (float)q[j + 1] * xb[j + 1];
            s2 += (float)q[j + 2] * xb[j + 2];
            s3 += (float)q[j + 3] * xb[j + 3];
        }
        sum += d * ((s0 + s1) + (s2 + s3));
        y += 34;
    }
    return sum;
}

// Q4_0 block: { f16 scale; uint8 q[16] } — low nibbles: elems 0..15, high: 16..31
float dot_q4_0(const uint8_t* y, const float* x, int64_t n) {
    float sum = 0.0f;
    const int64_t nblocks = n / 32;
    for (int64_t b = 0; b < nblocks; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, y, 2);
        const float d = f16_to_f32(d_bits);
        const uint8_t* q = y + 2;
        const float* xb = x + b * 32;
        float s0 = 0, s1 = 0;
        for (int j = 0; j < 16; j += 2) {
            s0 += (float)((int)(q[j + 0] & 0x0F) - 8) * xb[j + 0];
            s1 += (float)((int)(q[j + 1] & 0x0F) - 8) * xb[j + 1];
        }
        for (int j = 0; j < 16; j += 2) {
            s0 += (float)((int)(q[j + 0] >> 4) - 8) * xb[16 + j + 0];
            s1 += (float)((int)(q[j + 1] >> 4) - 8) * xb[16 + j + 1];
        }
        sum += d * (s0 + s1);
        y += 18;
    }
    return sum;
}

} // namespace

void kernels_scalar_fill(Kernels* k) {
    k->dot_f32 = dot_f32;
    k->dot_f16 = dot_f16;
    k->dot_bf16 = dot_bf16;
    k->dot_q8_0 = dot_q8_0;
    k->dot_q4_0 = dot_q4_0;
}

} // namespace haidass
