// generate.h — generation loop: prefill + decode with streaming output.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "model.h"
#include "sampler.h"
#include "thread_pool.h"

namespace haidass {

struct GenParams {
    int max_tokens = 256;
    float temp = 0.0f;      // <= 0 -> greedy (model default)
    int top_k = 40;
    float top_p = 0.9f;
    uint64_t seed = 42;
    int max_ctx = 2048;
};

struct GenStats {
    int prompt_tokens = 0;
    int gen_tokens = 0;
    double prefill_s = 0.0;
    double decode_s = 0.0;
};

class Generator {
public:
    // n_threads: workers for matmul parallelism.
    Generator(const Model& model, int n_threads);

    // Generate from a fully-templated prompt. on_piece receives decoded text
    // incrementally; returning false from it aborts generation.
    // Returns false (with err set) on hard errors.
    bool generate(const std::string& prompt, const GenParams& params,
                  const std::function<bool(const std::string&)>& on_piece,
                  GenStats* stats, std::string* err);

private:
    const Model& model_;
    ThreadPool tp_;
};

} // namespace haidass
