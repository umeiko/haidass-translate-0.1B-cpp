// kernels_avx2.cpp — AVX2+FMA+F16C dot kernels (own TU with ISA flags;
// installed only after a runtime cpuid check).
#include "kernels.h"

#if defined(HAIDASS_HAVE_AVX2)

#    include <cstring>
#    include <immintrin.h>

namespace haidass {
namespace {

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

float dot_f32(const float* y, const float* x, int64_t n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(x + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(y + i + 8), _mm256_loadu_ps(x + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(y + i), _mm256_loadu_ps(x + i), acc0);
    }
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) sum += y[i] * x[i];
    return sum;
}

float dot_f16(const uint16_t* y, const float* x, int64_t n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 y0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(y + i)));
        __m256 y1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(y + i + 8)));
        acc0 = _mm256_fmadd_ps(y0, _mm256_loadu_ps(x + i), acc0);
        acc1 = _mm256_fmadd_ps(y1, _mm256_loadu_ps(x + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 y0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(y + i)));
        acc0 = _mm256_fmadd_ps(y0, _mm256_loadu_ps(x + i), acc0);
    }
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) sum += f16_to_f32(y[i]) * x[i];
    return sum;
}

float dot_bf16(const uint16_t* y, const float* x, int64_t n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i y0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(y + i))), 16);
        __m256i y1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(y + i + 8))), 16);
        acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(y0), _mm256_loadu_ps(x + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_castsi256_ps(y1), _mm256_loadu_ps(x + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256i y0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(y + i))), 16);
        acc0 = _mm256_fmadd_ps(_mm256_castsi256_ps(y0), _mm256_loadu_ps(x + i), acc0);
    }
    float sum = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) sum += bf16_to_f32(y[i]) * x[i];
    return sum;
}

// Q8_0 block: { f16 scale; int8 q[32] }
float dot_q8_0(const uint8_t* y, const float* x, int64_t n) {
    const int64_t nblocks = n / 32;
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < nblocks; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, y, 2);
        const __m256 d = _mm256_set1_ps(f16_to_f32(d_bits));
        const int8_t* q = (const int8_t*)(y + 2);
        const float* xb = x + b * 32;
        __m256 bs0 = _mm256_setzero_ps(), bs1 = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 16) {
            __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + j))));
            __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + j + 8))));
            bs0 = _mm256_fmadd_ps(q0, _mm256_loadu_ps(xb + j), bs0);
            bs1 = _mm256_fmadd_ps(q1, _mm256_loadu_ps(xb + j + 8), bs1);
        }
        acc = _mm256_fmadd_ps(d, _mm256_add_ps(bs0, bs1), acc);
        y += 34;
    }
    return hsum256(acc);
}

// Q4_0 block: { f16 scale; uint8 q[16] } — low nibbles: elems 0..15, high: 16..31
float dot_q4_0(const uint8_t* y, const float* x, int64_t n) {
    const int64_t nblocks = n / 32;
    const __m128i off = _mm_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (int64_t b = 0; b < nblocks; ++b) {
        uint16_t d_bits;
        std::memcpy(&d_bits, y, 2);
        const __m256 d = _mm256_set1_ps(f16_to_f32(d_bits));
        const uint8_t* q = y + 2;
        const float* xb = x + b * 32;

        __m128i bytes = _mm_loadu_si128((const __m128i*)q);
        __m128i lo = _mm_sub_epi8(_mm_and_si128(bytes, _mm_set1_epi8(0x0F)), off);
        __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F)), off);

        __m256 lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        __m256 lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        __m256 hi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        __m256 hi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));

        __m256 bs = _mm256_setzero_ps();
        bs = _mm256_fmadd_ps(lo0, _mm256_loadu_ps(xb + 0), bs);
        bs = _mm256_fmadd_ps(lo1, _mm256_loadu_ps(xb + 8), bs);
        bs = _mm256_fmadd_ps(hi0, _mm256_loadu_ps(xb + 16), bs);
        bs = _mm256_fmadd_ps(hi1, _mm256_loadu_ps(xb + 24), bs);
        acc = _mm256_fmadd_ps(d, bs, acc);
        y += 18;
    }
    return hsum256(acc);
}

} // namespace

void kernels_avx2_fill(Kernels* k) {
    k->dot_f32 = dot_f32;
    k->dot_f16 = dot_f16;
    k->dot_bf16 = dot_bf16;
    k->dot_q8_0 = dot_q8_0;
    k->dot_q4_0 = dot_q4_0;
}

} // namespace haidass

#endif  // HAIDASS_HAVE_AVX2
