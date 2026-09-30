// ops.cpp
#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels.h"

namespace haidass {

void rmsnorm(float* out, const float* x, const float* w, int64_t n, float eps) {
    float ss = 0.0f;
    for (int64_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

void softmax(float* x, int64_t n) {
    float mx = x[0];
    for (int64_t i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    const float inv = 1.0f / (float)sum;
    for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

void rope_head(float* v, int64_t head_dim, const float* cos_t, const float* sin_t) {
    const int64_t half = head_dim / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float v0 = v[i], v1 = v[i + half];
        v[i]        = v0 * cos_t[i] - v1 * sin_t[i];
        v[i + half] = v1 * cos_t[i] + v0 * sin_t[i];
    }
}

static float dot_row(const TensorView& W, int64_t row, const float* x, int64_t k) {
    const uint8_t* r = W.row(row);
    switch (W.dtype) {
        case DT_F32:  return g_kernels.dot_f32((const float*)r, x, k);
        case DT_F16:  return g_kernels.dot_f16((const uint16_t*)r, x, k);
        case DT_BF16: return g_kernels.dot_bf16((const uint16_t*)r, x, k);
        case DT_Q8_0: return g_kernels.dot_q8_0(r, x, k);
        case DT_Q4_0: return g_kernels.dot_q4_0(r, x, k);
        default: throw std::runtime_error("unsupported weight dtype in matmul");
    }
}

void matmul(float* out, const TensorView& W, const float* x, ThreadPool& tp) {
    const int64_t k = W.ne[0];
    const int64_t rows = W.n_dims >= 2 ? W.ne[1] : 1;
    tp.parallel_for(rows, [&](int64_t start, int64_t end) {
        for (int64_t r = start; r < end; ++r) out[r] = dot_row(W, r, x, k);
    });
}

void dequant_row(const TensorView& W, int64_t row, float* out, int64_t n) {
    const uint8_t* r = W.row(row);
    switch (W.dtype) {
        case DT_F32:
            std::memcpy(out, r, sizeof(float) * (size_t)n);
            return;
        case DT_F16: {
            const uint16_t* p = (const uint16_t*)r;
            for (int64_t i = 0; i < n; ++i) out[i] = f16_to_f32(p[i]);
            return;
        }
        case DT_BF16: {
            const uint16_t* p = (const uint16_t*)r;
            for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
            return;
        }
        case DT_Q8_0: {
            const int64_t nb = n / 32;
            for (int64_t b = 0; b < nb; ++b) {
                uint16_t db;
                std::memcpy(&db, r + b * 34, 2);
                const float d = f16_to_f32(db);
                const int8_t* q = (const int8_t*)(r + b * 34 + 2);
                for (int j = 0; j < 32; ++j) out[b * 32 + j] = (float)q[j] * d;
            }
            return;
        }
        case DT_Q4_0: {
            const int64_t nb = n / 32;
            for (int64_t b = 0; b < nb; ++b) {
                uint16_t db;
                std::memcpy(&db, r + b * 18, 2);
                const float d = f16_to_f32(db);
                const uint8_t* q = r + b * 18 + 2;
                for (int j = 0; j < 16; ++j) {
                    out[b * 32 + j]      = (float)((int)(q[j] & 0x0F) - 8) * d;
                    out[b * 32 + 16 + j] = (float)((int)(q[j] >> 4) - 8) * d;
                }
            }
            return;
        }
        default: throw std::runtime_error("unsupported weight dtype in dequant_row");
    }
}

} // namespace haidass
