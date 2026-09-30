// sampler.h — greedy / temperature+top-k+top-p sampling over logits.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace haidass {

class Sampler {
public:
    explicit Sampler(uint64_t seed) : rng_(seed) {}

    // temp <= 0 -> greedy argmax. Otherwise temperature -> top-k -> top-p.
    int32_t sample(const float* logits, int64_t n, float temp, int top_k, float top_p);

private:
    std::mt19937_64 rng_;
    std::vector<std::pair<float, int32_t>> scratch_;
};

} // namespace haidass
