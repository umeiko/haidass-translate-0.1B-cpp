// model.cpp — Qwen3 forward pass.
//
// Architecture (per config.json of DALabCommunity/Haidass-Translate-143M):
//   pre-norm RMSNorm; attention with GQA (9 Q / 3 KV), per-head QK RMSNorm,
//   full-dim RoPE (NeoX split-half, theta=100000), no biases;
//   SwiGLU MLP; tied input/output embeddings.
#include "model.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels.h"
#include "ops.h"

namespace haidass {

namespace {
std::string layer_name(int l, const char* suffix) {
    return "blk." + std::to_string(l) + "." + suffix;
}
} // namespace

const TensorView* Model::require(const std::string& name) const {
    const TensorView* t = gguf_.tensor(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

bool Model::open(const uint8_t* data, size_t size, std::string* err) {
    if (!gguf_.open(data, size, err)) return false;
    if (gguf_.arch() != "qwen3") {
        if (err) *err = "unsupported architecture: " + gguf_.arch() + " (expected qwen3)";
        return false;
    }

    cfg_.n_layers = (int)gguf_.get_u32("qwen3.block_count");
    cfg_.n_heads = (int)gguf_.get_u32("qwen3.attention.head_count");
    cfg_.n_kv_heads = (int)gguf_.get_u32("qwen3.attention.head_count_kv");
    cfg_.hidden = (int)gguf_.get_u32("qwen3.embedding_length");
    cfg_.intermediate = (int)gguf_.get_u32("qwen3.feed_forward_length");
    cfg_.head_dim = (int)gguf_.get_u32("qwen3.attention.key_length",
                                       cfg_.n_heads ? cfg_.hidden / cfg_.n_heads : 0);
    cfg_.ctx_train = (int)gguf_.get_u32("qwen3.context_length", 4096);
    cfg_.rope_theta = gguf_.get_f32("qwen3.rope.freq_base", 100000.0f);
    cfg_.rms_eps = gguf_.get_f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    if (cfg_.n_layers <= 0 || cfg_.n_heads <= 0 || cfg_.n_kv_heads <= 0 || cfg_.hidden <= 0 ||
        cfg_.intermediate <= 0 || cfg_.head_dim <= 0) {
        if (err) *err = "incomplete model hyperparameters in GGUF metadata";
        return false;
    }

    if (!tok_.load(gguf_, err)) return false;
    cfg_.vocab = tok_.n_tokens();
    cfg_.bos = tok_.bos();
    cfg_.eos = tok_.eos();

    try {
        embd_ = require("token_embd.weight");
        output_norm_ = require("output_norm.weight");
        lm_head_ = gguf_.tensor("output.weight");  // null when tied
        layers_.resize(cfg_.n_layers);
        for (int l = 0; l < cfg_.n_layers; ++l) {
            Layer& L = layers_[l];
            L.attn_norm = require(layer_name(l, "attn_norm.weight"));
            L.q = require(layer_name(l, "attn_q.weight"));
            L.k = require(layer_name(l, "attn_k.weight"));
            L.v = require(layer_name(l, "attn_v.weight"));
            L.o = require(layer_name(l, "attn_output.weight"));
            L.q_norm = require(layer_name(l, "attn_q_norm.weight"));
            L.k_norm = require(layer_name(l, "attn_k_norm.weight"));
            L.ffn_norm = require(layer_name(l, "ffn_norm.weight"));
            L.gate = require(layer_name(l, "ffn_gate.weight"));
            L.up = require(layer_name(l, "ffn_up.weight"));
            L.down = require(layer_name(l, "ffn_down.weight"));
        }
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }

    // sanity-check shapes
    if (embd_->ne[0] != cfg_.hidden || embd_->ne[1] != cfg_.vocab) {
        if (err) *err = "bad token_embd shape";
        return false;
    }
    if (layers_[0].q->ne[0] != cfg_.hidden ||
        layers_[0].q->ne[1] != (int64_t)cfg_.n_heads * cfg_.head_dim) {
        if (err) *err = "bad attn_q shape";
        return false;
    }

    // RoPE tables (NeoX split-half): freq_i = theta^(-2i/head_dim)
    rope_ctx_ = cfg_.ctx_train;
    const int half = cfg_.head_dim / 2;
    rope_cos_.resize((size_t)rope_ctx_ * half);
    rope_sin_.resize((size_t)rope_ctx_ * half);
    for (int pos = 0; pos < rope_ctx_; ++pos) {
        for (int i = 0; i < half; ++i) {
            const float freq = 1.0f / powf(cfg_.rope_theta, (float)(2 * i) / (float)cfg_.head_dim);
            const float angle = (float)pos * freq;
            rope_cos_[(size_t)pos * half + i] = cosf(angle);
            rope_sin_[(size_t)pos * half + i] = sinf(angle);
        }
    }
    return true;
}

Model::RunState Model::make_state(int max_ctx) const {
    RunState s;
    const int64_t kv_dim = (int64_t)cfg_.n_kv_heads * cfg_.head_dim;
    s.x.resize(cfg_.hidden);
    s.xb.resize(cfg_.hidden);
    s.xb2.resize(cfg_.hidden);
    s.q.resize((int64_t)cfg_.n_heads * cfg_.head_dim);
    s.k.resize(kv_dim);
    s.v.resize(kv_dim);
    s.attn_out.resize((int64_t)cfg_.n_heads * cfg_.head_dim);
    s.att.resize(max_ctx);
    s.hb.resize(cfg_.intermediate);
    s.hb2.resize(cfg_.intermediate);
    s.logits.resize(cfg_.vocab);
    s.kcache.resize((int64_t)cfg_.n_layers * max_ctx * kv_dim);
    s.vcache.resize((int64_t)cfg_.n_layers * max_ctx * kv_dim);
    s.max_ctx = max_ctx;
    s.pos = 0;
    return s;
}

const float* Model::forward(RunState& s, int32_t token, int pos, ThreadPool& tp) const {
    const int64_t hidden = cfg_.hidden;
    const int64_t q_dim = (int64_t)cfg_.n_heads * cfg_.head_dim;
    const int64_t kv_dim = (int64_t)cfg_.n_kv_heads * cfg_.head_dim;
    const int64_t head_dim = cfg_.head_dim;
    const int64_t half = head_dim / 2;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const float* cos_t = rope_cos_.data() + (size_t)pos * half;
    const float* sin_t = rope_sin_.data() + (size_t)pos * half;

    // embedding lookup
    dequant_row(*embd_, token, s.x.data(), hidden);

    std::vector<float> nbuf(head_dim);  // qk-norm weight scratch
    for (int l = 0; l < cfg_.n_layers; ++l) {
        const Layer& L = layers_[l];

        // attention input norm + qkv projections
        dequant_row(*L.attn_norm, 0, s.xb2.data(), hidden);
        rmsnorm(s.xb.data(), s.x.data(), s.xb2.data(), hidden, cfg_.rms_eps);
        matmul(s.q.data(), *L.q, s.xb.data(), tp);
        matmul(s.k.data(), *L.k, s.xb.data(), tp);
        matmul(s.v.data(), *L.v, s.xb.data(), tp);

        // QK norm (per head, head_dim-wide RMSNorm) then RoPE
        dequant_row(*L.q_norm, 0, nbuf.data(), head_dim);
        for (int h = 0; h < cfg_.n_heads; ++h) {
            rmsnorm(s.q.data() + (int64_t)h * head_dim, s.q.data() + (int64_t)h * head_dim,
                    nbuf.data(), head_dim, cfg_.rms_eps);
            rope_head(s.q.data() + (int64_t)h * head_dim, head_dim, cos_t, sin_t);
        }
        dequant_row(*L.k_norm, 0, nbuf.data(), head_dim);
        for (int h = 0; h < cfg_.n_kv_heads; ++h) {
            rmsnorm(s.k.data() + (int64_t)h * head_dim, s.k.data() + (int64_t)h * head_dim,
                    nbuf.data(), head_dim, cfg_.rms_eps);
            rope_head(s.k.data() + (int64_t)h * head_dim, head_dim, cos_t, sin_t);
        }

        // KV cache store
        const int64_t layer_stride = (int64_t)s.max_ctx * kv_dim;
        float* kc = s.kcache.data() + (int64_t)l * layer_stride + (int64_t)pos * kv_dim;
        float* vc = s.vcache.data() + (int64_t)l * layer_stride + (int64_t)pos * kv_dim;
        std::memcpy(kc, s.k.data(), sizeof(float) * kv_dim);
        std::memcpy(vc, s.v.data(), sizeof(float) * kv_dim);

        // multi-head attention with GQA
        const int group = cfg_.n_heads / cfg_.n_kv_heads;
        for (int h = 0; h < cfg_.n_heads; ++h) {
            const int kvh = h / group;
            const float* qh = s.q.data() + (int64_t)h * head_dim;
            const float* kbase = s.kcache.data() + (int64_t)l * layer_stride + (int64_t)kvh * head_dim;
            const float* vbase = s.vcache.data() + (int64_t)l * layer_stride + (int64_t)kvh * head_dim;
            for (int t = 0; t <= pos; ++t) {
                s.att[t] = g_kernels.dot_f32(kbase + (int64_t)t * kv_dim, qh, head_dim) * scale;
            }
            softmax(s.att.data(), pos + 1);
            float* oh = s.attn_out.data() + (int64_t)h * head_dim;
            std::memset(oh, 0, sizeof(float) * head_dim);
            for (int t = 0; t <= pos; ++t) {
                const float w = s.att[t];
                const float* vh = vbase + (int64_t)t * kv_dim;
                for (int64_t d = 0; d < head_dim; ++d) oh[d] += w * vh[d];
            }
        }

        // output projection + residual
        matmul(s.xb2.data(), *L.o, s.attn_out.data(), tp);
        for (int64_t i = 0; i < hidden; ++i) s.x[i] += s.xb2[i];

        // MLP (SwiGLU)
        dequant_row(*L.ffn_norm, 0, s.xb2.data(), hidden);
        rmsnorm(s.xb.data(), s.x.data(), s.xb2.data(), hidden, cfg_.rms_eps);
        matmul(s.hb.data(), *L.gate, s.xb.data(), tp);
        matmul(s.hb2.data(), *L.up, s.xb.data(), tp);
        for (int64_t i = 0; i < cfg_.intermediate; ++i) s.hb[i] = silu(s.hb[i]) * s.hb2[i];
        matmul(s.xb2.data(), *L.down, s.hb.data(), tp);
        for (int64_t i = 0; i < hidden; ++i) s.x[i] += s.xb2[i];
    }

    // final norm + logits
    dequant_row(*output_norm_, 0, s.xb2.data(), hidden);
    rmsnorm(s.xb.data(), s.x.data(), s.xb2.data(), hidden, cfg_.rms_eps);
    matmul(s.logits.data(), lm_head_ ? *lm_head_ : *embd_, s.xb.data(), tp);
    return s.logits.data();
}

std::string Model::weight_summary() const {
    std::string out = "arch=qwen3 layers=" + std::to_string(cfg_.n_layers) +
                      " hidden=" + std::to_string(cfg_.hidden) +
                      " heads=" + std::to_string(cfg_.n_heads) +
                      " kv_heads=" + std::to_string(cfg_.n_kv_heads) +
                      " head_dim=" + std::to_string(cfg_.head_dim) +
                      " vocab=" + std::to_string(cfg_.vocab);
    out += " embd=";
    out += dtype_name(embd_->dtype);
    out += " blk0.q=";
    out += dtype_name(layers_[0].q->dtype);
    return out;
}

} // namespace haidass
