// sampler.cpp
#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace haidass {

int32_t Sampler::sample(const float* logits, int64_t n, float temp, int top_k, float top_p) {
    if (temp <= 0.0f) {
        return (int32_t)(std::max_element(logits, logits + n) - logits);
    }

    // temperature
    scratch_.clear();
    scratch_.reserve((size_t)n);
    for (int64_t i = 0; i < n; ++i) scratch_.emplace_back(logits[i] / temp, (int32_t)i);

    // top-k (partial sort)
    int64_t keep = n;
    if (top_k > 0 && top_k < n) {
        std::partial_sort(scratch_.begin(), scratch_.begin() + top_k, scratch_.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        keep = top_k;
    } else {
        std::sort(scratch_.begin(), scratch_.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
    }

    // softmax over kept
    const float mx = scratch_[0].first;
    double sum = 0.0;
    for (int64_t i = 0; i < keep; ++i) {
        scratch_[i].first = expf(scratch_[i].first - mx);
        sum += scratch_[i].first;
    }

    // top-p (nucleus) on cumulative probability
    if (top_p > 0.0f && top_p < 1.0f) {
        double cum = 0.0;
        int64_t last = keep;
        for (int64_t i = 0; i < keep; ++i) {
            cum += scratch_[i].first / sum;
            if (cum >= top_p) {
                last = i + 1;
                break;
            }
        }
        keep = last;
        sum = 0.0;
        for (int64_t i = 0; i < keep; ++i) sum += scratch_[i].first;
    }

    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng_);
    for (int64_t i = 0; i < keep; ++i) {
        r -= scratch_[i].first;
        if (r <= 0.0) return scratch_[i].second;
    }
    return scratch_[keep - 1].second;
}

} // namespace haidass
