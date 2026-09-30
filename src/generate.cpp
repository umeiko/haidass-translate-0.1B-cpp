// generate.cpp
#include "generate.h"

#include <chrono>

namespace haidass {

namespace {

// Buffers partial UTF-8 byte sequences (byte-fallback tokens can split them)
// and only emits complete sequences.
class Utf8Streamer {
public:
    // Returns the bytes safe to emit now.
    std::string push(const std::string& piece) {
        buf_ += piece;
        size_t emit = 0;
        size_t i = 0;
        while (i < buf_.size()) {
            const uint8_t c = (uint8_t)buf_[i];
            size_t len = 1;
            if ((c & 0x80) == 0) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            else { ++i; emit = i; continue; }  // stray byte: emit as-is
            if (i + len > buf_.size()) break;  // incomplete: wait
            i += len;
            emit = i;
        }
        std::string out = buf_.substr(0, emit);
        buf_.erase(0, emit);
        return out;
    }
    std::string flush() { std::string out = buf_; buf_.clear(); return out; }

private:
    std::string buf_;
};

} // namespace

Generator::Generator(const Model& model, int n_threads) : model_(model), tp_(n_threads) {}

bool Generator::generate(const std::string& prompt, const GenParams& params,
                         const std::function<bool(const std::string&)>& on_piece,
                         GenStats* stats, std::string* err) {
    const Tokenizer& tok = model_.tokenizer();
    std::vector<int32_t> ids = tok.encode(prompt, /*parse_special=*/true);
    if (ids.empty()) {
        if (err) *err = "empty tokenized prompt";
        return false;
    }
    if ((int)ids.size() >= params.max_ctx - 1) {
        if (err) *err = "prompt too long for context";
        return false;
    }

    Model::RunState state = model_.make_state(params.max_ctx);
    Sampler sampler(params.seed);
    Utf8Streamer streamer;

    const auto t0 = std::chrono::steady_clock::now();

    const float* logits = nullptr;
    for (size_t i = 0; i < ids.size(); ++i) {
        logits = model_.forward(state, ids[i], (int)i, tp_);
    }
    int pos = (int)ids.size();

    const auto t1 = std::chrono::steady_clock::now();

    int gen = 0;
    for (int t = 0; t < params.max_tokens; ++t) {
        const int32_t next = sampler.sample(logits, model_.config().vocab, params.temp,
                                            params.top_k, params.top_p);
        if (tok.is_eog(next)) break;
        ++gen;
        std::string text = streamer.push(tok.piece(next));
        if (!text.empty() && on_piece && !on_piece(text)) break;
        if (pos >= params.max_ctx - 1) break;
        logits = model_.forward(state, next, pos++, tp_);
    }
    std::string rest = streamer.flush();
    if (!rest.empty() && on_piece) on_piece(rest);

    const auto t2 = std::chrono::steady_clock::now();
    if (stats) {
        stats->prompt_tokens = (int)ids.size();
        stats->gen_tokens = gen;
        stats->prefill_s = std::chrono::duration<double>(t1 - t0).count();
        stats->decode_s = std::chrono::duration<double>(t2 - t1).count();
    }
    return true;
}

} // namespace haidass
