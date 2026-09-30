// kernels_neon.cpp — ARM NEON dot kernels (own TU; runtime-dispatched on
// armhf via HWCAP check, always-on for arm64).
//
// Portability notes: uses vmlaq (not vfmaq) and avoids f16/bf16 arithmetic
// intrinsics so it compiles for ARMv7 hard-float as well as arm64.
#include "kernels.h"

#if defined(HAIDASS_HAVE_NEON)

#    include <cstring>
#    include <arm_neon.h>

namespace haidass {
namespace {

inline float hsumq(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}

float dot_f32(const float* y, const float* x, int64_t n) {
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vmlaq_f32(acc0, vld1q_f32(y + i), vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vld1q_f32(y + i + 4), vld1q_f32(x + i + 4));
    }
    for (; i + 4 <= n; i += 4) acc0 = vmlaq_f32(acc0, vld1q_f32(y + i), vld1q_f32(x + i));
    float sum = hsumq(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) sum += y[i] * x[i];
    return sum;
}

float dot_f16(const uint16_t* y, const float* x, int64_t n) {
#    if defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vmlaq_f32(acc0, vcvt_f32_f16(vld1_f16((const float16_t*)(y + i))), vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vcvt_f32_f16(vld1_f16((const float16_t*)(y + i + 4))), vld1q_f32(x + i + 4));
    }
    for (; i + 4 <= n; i += 4) acc0 = vmlaq_f32(acc0, vcvt_f32_f16(vld1_f16((const float16_t*)(y + i))), vld1q_f32(x + i));
    float sum = hsumq(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) sum += f16_to_f32(y[i]) * x[i];
    return sum;
#    else
    // armv7 has no f16 SIMD convert instruction with plain -mfpu=neon.
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += f16_to_f32(y[i]) * x[i];
    return sum;
#    endif
}

float dot_bf16(const uint16_t* y, const float* x, int64_t n) {
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint32x4_t y0 = vshll_n_u16(vld1_u16(y + i), 16);
        uint32x4_t y1 = vshll_n_u16(vld1_u16(y + i + 4), 16);
        acc0 = vmlaq_f32(acc0, vreinterpretq_f32_u32(y0), vld1q_f32(x + i));
        acc1 = vmlaq_f32(acc1, vreinterpretq_f32_u32(y1), vld1q_f32(x + i + 4));
    }
    for (; i + 4 <= n; i += 4) {
        uint32x4_t y0 = vshll_n_u16(vld1_u16(y + i), 16);
        acc0 = vmlaq_f32(acc0, vreinterpretq_f32_u32(y0), vld1q_f32(x + i));
    }
    float sum = hsumq(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) sum += bf16_to_f32(y[i]) * x[i];
    return sum;
}

// widen 8x i8 -> two f32x4
inline void dot8_i8(int8x8_t q, const float* x, float32x4_t& acc0, float32x4_t& acc1) {
    int16x8_t w = vmovl_s8(q);
    acc0 = vmlaq_f32(acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), vld1q_f32(x));
    acc1 = vmlaq_f32(acc1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w))), vld1q_f32(x + 4));
}

float dot_q8_0(const uint8_t* y, const float* x, int64_t n) {
    const int64_t nblocks = n / 32;
    float32x4_t acc = vdupq_n_f32(0);
    for (int64_t b = 0; b < nblocks; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, y, 2);
        const float32x4_t d = vdupq_n_f32(f16_to_f32(d_bits));
        const int8_t* q = (const int8_t*)(y + 2);
        const float* xb = x + b * 32;
        int8x16_t q0 = vld1q_s8(q), q1 = vld1q_s8(q + 16);
        float32x4_t bs0 = vdupq_n_f32(0), bs1 = vdupq_n_f32(0), bs2 = vdupq_n_f32(0), bs3 = vdupq_n_f32(0);
        dot8_i8(vget_low_s8(q0), xb + 0, bs0, bs1);
        dot8_i8(vget_high_s8(q0), xb + 8, bs2, bs3);
        bs0 = vaddq_f32(vaddq_f32(bs0, bs1), vaddq_f32(bs2, bs3));
        // dot8_i8 accumulates into its outputs — zero before reuse
        bs1 = vdupq_n_f32(0);
        bs2 = vdupq_n_f32(0);
        dot8_i8(vget_low_s8(q1), xb + 16, bs1, bs2);
        float32x4_t bs4 = vdupq_n_f32(0), bs5 = vdupq_n_f32(0);
        dot8_i8(vget_high_s8(q1), xb + 24, bs4, bs5);
        bs1 = vaddq_f32(vaddq_f32(bs1, bs2), vaddq_f32(bs4, bs5));
        acc = vmlaq_f32(acc, d, vaddq_f32(bs0, bs1));
        y += 34;
    }
    return hsumq(acc);
}

float dot_q4_0(const uint8_t* y, const float* x, int64_t n) {
    const int64_t nblocks = n / 32;
    const int8x16_t off = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0);
    for (int64_t b = 0; b < nblocks; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, y, 2);
        const float32x4_t d = vdupq_n_f32(f16_to_f32(d_bits));
        const uint8_t* q = y + 2;
        const float* xb = x + b * 32;
        uint8x16_t bytes = vld1q_u8(q);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, vdupq_n_u8(0x0F))), off);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4)), off);
        float32x4_t bs0 = vdupq_n_f32(0), bs1 = vdupq_n_f32(0), bs2 = vdupq_n_f32(0), bs3 = vdupq_n_f32(0);
        dot8_i8(vget_low_s8(lo), xb + 0, bs0, bs1);
        dot8_i8(vget_high_s8(lo), xb + 8, bs2, bs3);
        bs0 = vaddq_f32(vaddq_f32(bs0, bs1), vaddq_f32(bs2, bs3));
        // dot8_i8 accumulates into its outputs — zero before reuse
        bs1 = vdupq_n_f32(0);
        bs2 = vdupq_n_f32(0);
        dot8_i8(vget_low_s8(hi), xb + 16, bs1, bs2);
        float32x4_t bs4 = vdupq_n_f32(0), bs5 = vdupq_n_f32(0);
        dot8_i8(vget_high_s8(hi), xb + 24, bs4, bs5);
        bs1 = vaddq_f32(vaddq_f32(bs1, bs2), vaddq_f32(bs4, bs5));
        acc = vmlaq_f32(acc, d, vaddq_f32(bs0, bs1));
        y += 18;
    }
    return hsumq(acc);
}

} // namespace

void kernels_neon_fill(Kernels* k) {
    k->dot_f32 = dot_f32;
#    if defined(__aarch64__)
    k->dot_f16 = dot_f16;
#    endif
    k->dot_bf16 = dot_bf16;
    k->dot_q8_0 = dot_q8_0;
    k->dot_q4_0 = dot_q4_0;
}

} // namespace haidass

#endif  // HAIDASS_HAVE_NEON
