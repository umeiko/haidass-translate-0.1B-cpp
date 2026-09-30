// ops.h — math ops over f32 activations + (possibly quantized) weights.
#pragma once

#include <cmath>
#include <cstdint>

#include "tensor.h"
#include "thread_pool.h"

namespace haidass {

// out = x * rsqrt(mean(x^2) + eps) * w
void rmsnorm(float* out, const float* x, const float* w, int64_t n, float eps);

// in-place softmax with max subtraction
void softmax(float* x, int64_t n);

// Apply rotary position embedding (split-half / NeoX layout) to one head.
// cos_t/sin_t point at the row for the current position, size head_dim/2.
void rope_head(float* v, int64_t head_dim, const float* cos_t, const float* sin_t);

inline float silu(float x) { return x / (1.0f + expf(-x)); }

// out[rows] = W[rows x k] @ x[k]. W rows may be f32/f16/bf16/q8_0/q4_0.
// Parallelized over rows via the thread pool.
void matmul(float* out, const TensorView& W, const float* x, ThreadPool& tp);

// Dequantize one row of `n` elements into f32 (used for embedding lookup).
void dequant_row(const TensorView& W, int64_t row, float* out, int64_t n);

} // namespace haidass
