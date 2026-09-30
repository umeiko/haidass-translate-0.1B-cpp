// gguf_test.cpp — parse the fixture GGUF and validate structure/metadata.
#include <cstdint>
#include <string>

#include "gguf.h"
#include "platform.h"
#include "test_util.h"

using namespace haidass;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gguf_test <fixture.gguf>\n");
        return 2;
    }
    MappedFile mf;
    std::string err;
    CHECK(mf.open(argv[1], &err));
    if (!mf.is_open()) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return test_summary("gguf_test");
    }

    GgufFile g;
    CHECK(g.open(mf.data(), mf.size(), &err));
    if (g_failures) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return test_summary("gguf_test");
    }

    CHECK(g.version() == 3);
    CHECK(g.arch() == "qwen3");
    CHECK(g.get_u32("qwen3.block_count") == 2);
    CHECK(g.get_u32("qwen3.embedding_length") == 64);
    CHECK(g.get_u32("qwen3.attention.head_count") == 4);
    CHECK(g.get_u32("qwen3.attention.head_count_kv") == 2);
    CHECK(g.get_u32("qwen3.attention.key_length") == 16);
    CHECK_NEAR(g.get_f32("qwen3.rope.freq_base"), 100000.0, 1e-3);
    CHECK(g.get_str("tokenizer.ggml.model") == "llama");
    CHECK(g.get_u32("tokenizer.ggml.eos_token_id") == 5);

    // 2 layers x 11 tensors + token_embd + output_norm
    CHECK(g.num_tensors() == 24);

    const TensorView* embd = g.tensor("token_embd.weight");
    CHECK(embd != nullptr);
    if (embd) {
        CHECK(embd->dtype == DT_F32);
        CHECK(embd->n_dims == 2);
        CHECK(embd->ne[0] == 64);
        CHECK(embd->ne[1] == 64000);
        CHECK((reinterpret_cast<uintptr_t>(embd->data) % 32) == 0);
        CHECK(embd->total_bytes() == 64 * 64000 * 4);
    }
    const TensorView* q0 = g.tensor("blk.0.attn_q.weight");
    CHECK(q0 != nullptr);
    if (q0) {
        CHECK(q0->ne[0] == 64);
        CHECK(q0->ne[1] == 64);
    }
    CHECK(g.tensor("does_not_exist") == nullptr);

    return test_summary("gguf_test");
}
