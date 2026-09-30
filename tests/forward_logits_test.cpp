// forward_logits_test.cpp — C++ forward pass vs numpy reference logits.
//
// logits_ref.bin layout: u32 n_ids | u32 vocab | i32 ids[n_ids] | f32 logits[vocab]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "kernels.h"
#include "model.h"
#include "platform.h"
#include "test_util.h"
#include "thread_pool.h"

using namespace haidass;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: forward_logits_test <fixture.gguf> <logits_ref.bin>\n");
        return 2;
    }
    kernels_init();

    MappedFile mf;
    std::string err;
    CHECK(mf.open(argv[1], &err));
    if (!mf.is_open()) return test_summary("forward_logits_test");

    Model model;
    CHECK(model.open(mf.data(), mf.size(), &err));
    if (g_failures) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return test_summary("forward_logits_test");
    }

    std::ifstream in(argv[2], std::ios::binary);
    CHECK(in.good());
    uint32_t n_ids = 0, vocab = 0;
    in.read((char*)&n_ids, 4);
    in.read((char*)&vocab, 4);
    CHECK(n_ids > 0 && vocab == (uint32_t)model.config().vocab);
    std::vector<int32_t> ids(n_ids);
    in.read((char*)ids.data(), 4 * n_ids);
    std::vector<float> ref(vocab);
    in.read((char*)ref.data(), 4 * vocab);
    CHECK(in.good());

    ThreadPool tp(1);  // single-threaded for a clean reference comparison
    Model::RunState state = model.make_state(512);
    const float* logits = nullptr;
    for (uint32_t i = 0; i < n_ids; ++i) logits = model.forward(state, ids[i], (int)i, tp);
    CHECK(logits != nullptr);

    double max_abs = 0, max_rel = 0;
    double dot = 0, na = 0, nb = 0;
    for (uint32_t i = 0; i < vocab; ++i) {
        const double d = std::fabs((double)logits[i] - ref[i]);
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / (std::fabs((double)ref[i]) + 1e-6));
        dot += (double)logits[i] * ref[i];
        na += (double)logits[i] * logits[i];
        nb += (double)ref[i] * ref[i];
    }
    const double cosine = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const int argmax_got = (int)(std::max_element(logits, logits + vocab) - logits);
    const int argmax_ref = (int)(std::max_element(ref.begin(), ref.end()) - ref.begin());

    std::fprintf(stderr, "logits: max_abs=%.3g max_rel=%.3g cosine=%.6f argmax got=%d ref=%d\n",
                 max_abs, max_rel, cosine, argmax_got, argmax_ref);

    CHECK(argmax_got == argmax_ref);
    CHECK(cosine > 0.9999);
    CHECK(max_abs < 0.05);

    return test_summary("forward_logits_test");
}
