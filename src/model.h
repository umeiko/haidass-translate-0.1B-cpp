// model.h — Qwen3 (LLM-arch) model: weights + forward pass + KV cache.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"
#include "tensor.h"
#include "thread_pool.h"
#include "tokenizer.h"

namespace haidass {

struct Config {
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int hidden = 0;
    int intermediate = 0;
    int vocab = 0;
    int ctx_train = 0;
    float rope_theta = 100000.0f;
    float rms_eps = 1e-6f;
    int32_t bos = -1;
    int32_t eos = -1;
};

class Model {
public:
    // Parse from an in-memory GGUF image (mmap or embedded). The caller keeps
    // the buffer alive (Model stores pointers into it).
    bool open(const uint8_t* data, size_t size, std::string* err);

    const Config& config() const { return cfg_; }
    const Tokenizer& tokenizer() const { return tok_; }

    // Per-run state: activations + KV cache. Allocate once per generation.
    struct RunState {
        std::vector<float> x;        // hidden
        std::vector<float> xb;       // hidden (normed / residual scratch)
        std::vector<float> xb2;      // hidden (projection output)
        std::vector<float> q;        // n_heads * head_dim
        std::vector<float> k, v;     // n_kv_heads * head_dim
        std::vector<float> attn_out; // n_heads * head_dim
        std::vector<float> att;      // max_ctx scores for current head
        std::vector<float> hb, hb2;  // intermediate
        std::vector<float> logits;   // vocab
        std::vector<float> kcache;   // n_layers * max_ctx * kv_dim
        std::vector<float> vcache;
        int max_ctx = 0;
        int pos = 0;
    };

    RunState make_state(int max_ctx) const;

    // Run one token at position `pos`; returns pointer to logits (vocab size).
    const float* forward(RunState& s, int32_t token, int pos, ThreadPool& tp) const;

    // Human-readable summary of tensor dtypes (for diagnostics/tests).
    std::string weight_summary() const;

private:
    const TensorView* require(const std::string& name) const;

    GgufFile gguf_;
    Tokenizer tok_;
    Config cfg_;

    const TensorView* embd_ = nullptr;
    const TensorView* output_norm_ = nullptr;
    const TensorView* lm_head_ = nullptr;  // falls back to embd_ when tied

    struct Layer {
        const TensorView *attn_norm, *q, *k, *v, *o, *q_norm, *k_norm;
        const TensorView *ffn_norm, *gate, *up, *down;
    };
    std::vector<Layer> layers_;

    std::vector<float> rope_cos_, rope_sin_;  // [ctx_train][head_dim/2]
    int rope_ctx_ = 0;
};

} // namespace haidass
