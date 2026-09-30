// ops_test.cpp — kernel/ops correctness: SIMD path vs scalar reference math.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "kernels.h"
#include "ops.h"
#include "test_util.h"
#include "thread_pool.h"

using namespace haidass;

namespace {

std::vector<uint8_t> quant_q8_0(const std::vector<float>& v) {
    const int64_t n = (int64_t)v.size();
    std::vector<uint8_t> out((size_t)(n / 32) * 34);
    uint8_t* p = out.data();
    for (int64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(v[b * 32 + j]));
        const float d = amax / 127.0f;
        const uint16_t db = f32_to_f16(d);
        std::memcpy(p, &db, 2);
        const float d32 = f16_to_f32(db);
        for (int j = 0; j < 32; ++j) {
            const float q = d32 > 0 ? v[b * 32 + j] / d32 : 0.0f;
            int r = (int)std::lroundf(q);
            r = std::max(-127, std::min(127, r));
            p[2 + j] = (uint8_t)(int8_t)r;
        }
        p += 34;
    }
    return out;
}

std::vector<uint8_t> quant_q4_0(const std::vector<float>& v) {
    const int64_t n = (int64_t)v.size();
    std::vector<uint8_t> out((size_t)(n / 32) * 18);
    uint8_t* p = out.data();
    for (int64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(v[b * 32 + j]));
        const float d = amax / 7.0f;
        const uint16_t db = f32_to_f16(d);
        std::memcpy(p, &db, 2);
        const float d32 = f16_to_f32(db);
        for (int j = 0; j < 16; ++j) {
            const int lo = std::max(0, std::min(15, (int)std::lroundf(d32 > 0 ? v[b * 32 + j] / d32 : 0.0f) + 8));
            const int hi = std::max(0, std::min(15, (int)std::lroundf(d32 > 0 ? v[b * 32 + 16 + j] / d32 : 0.0f) + 8));
            p[2 + j] = (uint8_t)(lo | (hi << 4));
        }
        p += 18;
    }
    return out;
}

std::vector<float> dequant_q8_0(const std::vector<uint8_t>& q, int64_t n) {
    std::vector<float> out((size_t)n);
    for (int64_t b = 0; b < n / 32; ++b) {
        uint16_t db;
        std::memcpy(&db, q.data() + b * 34, 2);
        const float d = f16_to_f32(db);
        for (int j = 0; j < 32; ++j)
            out[b * 32 + j] = d * (float)(int8_t)q[b * 34 + 2 + j];
    }
    return out;
}

std::vector<float> dequant_q4_0(const std::vector<uint8_t>& q, int64_t n) {
    std::vector<float> out((size_t)n);
    for (int64_t b = 0; b < n / 32; ++b) {
        uint16_t db;
        std::memcpy(&db, q.data() + b * 18, 2);
        const float d = f16_to_f32(db);
        for (int j = 0; j < 16; ++j) {
            out[b * 32 + j] = d * (float)((int)(q[b * 18 + 2 + j] & 0x0F) - 8);
            out[b * 32 + 16 + j] = d * (float)((int)(q[b * 18 + 2 + j] >> 4) - 8);
        }
    }
    return out;
}

float naive_dot(const float* a, const float* b, int64_t n) {
    double s = 0;
    for (int64_t i = 0; i < n; ++i) s += (double)a[i] * b[i];
    return (float)s;
}

void check_dots(const Kernels& k, const char* tier) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t n = 1024;
    std::vector<float> x(n), y(n);
    for (int64_t i = 0; i < n; ++i) { x[i] = nd(rng); y[i] = nd(rng); }

    // f32
    CHECK_NEAR(k.dot_f32(y.data(), x.data(), n), naive_dot(y.data(), x.data(), n), 1e-2);
    // f16 / bf16
    std::vector<uint16_t> y16(n), yb(n);
    std::vector<float> y16f(n), ybf(n);
    for (int64_t i = 0; i < n; ++i) {
        y16[i] = f32_to_f16(y[i]);
        y16f[i] = f16_to_f32(y16[i]);
        uint32_t bits;
        std::memcpy(&bits, &y[i], 4);
        yb[i] = (uint16_t)(bits >> 16);
        ybf[i] = bf16_to_f32(yb[i]);
    }
    CHECK_NEAR(k.dot_f16(y16.data(), x.data(), n), naive_dot(y16f.data(), x.data(), n), 1e-1);
    CHECK_NEAR(k.dot_bf16(yb.data(), x.data(), n), naive_dot(ybf.data(), x.data(), n), 1.0);
    // q8_0 / q4_0
    const std::vector<uint8_t> q8 = quant_q8_0(y);
    const std::vector<float> dq8 = dequant_q8_0(q8, n);
    CHECK_NEAR(k.dot_q8_0(q8.data(), x.data(), n), naive_dot(dq8.data(), x.data(), n), 1e-2);
    const std::vector<uint8_t> q4 = quant_q4_0(y);
    const std::vector<float> dq4 = dequant_q4_0(q4, n);
    CHECK_NEAR(k.dot_q4_0(q4.data(), x.data(), n), naive_dot(dq4.data(), x.data(), n), 1e-2);
    (void)tier;
}

} // namespace

int main() {
    kernels_init();

    // scalar table is the reference
    Kernels scalar;
    kernels_scalar_fill(&scalar);
    check_dots(scalar, "scalar");
    // active (possibly SIMD) table must agree with scalar
    check_dots(g_kernels, kernels_active_tier());

    // rmsnorm
    {
        std::vector<float> x = {1, -2, 3, 0.5f}, w = {2, 1, 0.5f, 1}, out(4);
        rmsnorm(out.data(), x.data(), w.data(), 4, 1e-6f);
        const float ss = (1 + 4 + 9 + 0.25f) / 4.0f;
        const float scale = 1.0f / std::sqrt(ss + 1e-6f);
        for (int i = 0; i < 4; ++i) CHECK_NEAR(out[i], x[i] * scale * w[i], 1e-5);
    }
    // softmax
    {
        std::vector<float> x = {1, 2, 3, 4};
        softmax(x.data(), 4);
        double s = 0;
        for (float v : x) s += v;
        CHECK_NEAR(s, 1.0, 1e-5);
        CHECK(x[3] > x[2] && x[2] > x[1] && x[1] > x[0]);
    }
    // rope preserves norm (rotation is orthogonal)
    {
        std::vector<float> v = {1, 2, 3, 4}, orig = v;
        std::vector<float> cos_t = {0.9f, 0.8f}, sin_t = {0.43589f, 0.6f};
        rope_head(v.data(), 4, cos_t.data(), sin_t.data());
        // (0.9^2+0.43589^2) == 1, (0.8^2+0.6^2) == 1
        float n0 = 0, n1 = 0;
        for (int i = 0; i < 4; ++i) { n0 += v[i] * v[i]; n1 += orig[i] * orig[i]; }
        CHECK_NEAR(n0, n1, 1e-3);
        // rotate twice with inverse angle restores original
        rope_head(v.data(), 4, cos_t.data(), sin_t.data());  // not inverse; skip
        (void)v;
    }
    // matmul via TensorView (f32)
    {
        const int64_t rows = 8, k = 64;
        std::vector<float> W(rows * k), x(k), out(rows);
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : W) v = nd(rng);
        for (auto& v : x) v = nd(rng);
        TensorView tv;
        tv.dtype = DT_F32;
        tv.n_dims = 2;
        tv.ne[0] = k;
        tv.ne[1] = rows;
        tv.data = (const uint8_t*)W.data();
        ThreadPool tp(4);
        matmul(out.data(), tv, x.data(), tp);
        for (int64_t r = 0; r < rows; ++r)
            CHECK_NEAR(out[r], naive_dot(W.data() + r * k, x.data(), k), 1e-2);
    }

    return test_summary("ops_test");
}
