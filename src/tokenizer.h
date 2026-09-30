// tokenizer.h — SentencePiece-style tokenizer over the GGUF embedded vocab.
//
// Semantics replicate llama.cpp's "llama" (LLAMA_VOCAB_TYPE_SPM) path:
//   fragment split by special tokens -> optional " " prefix -> " " => U+2581
//   -> greedy bigram merge by piece score -> byte fallback for unknown spans.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "gguf.h"

namespace haidass {

class Tokenizer {
public:
    bool load(const GgufFile& gguf, std::string* err);

    // Encode text. parse_special=true treats occurrences of special token
    // strings (e.g. "<|im_start|>") as those tokens.
    std::vector<int32_t> encode(const std::string& text, bool parse_special = true) const;

    // Decode a token sequence to text. special=false skips control/unknown
    // tokens (user-defined specials are emitted verbatim, as in llama.cpp).
    std::string decode(const std::vector<int32_t>& tokens, bool special = false) const;

    // Render a single token as its piece (same skip rules as decode()).
    std::string piece(int32_t token, bool special = false) const;

    int32_t n_tokens() const { return static_cast<int32_t>(tokens_.size()); }
    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }

    bool is_eog(int32_t t) const { return t == eos_; }

    // Token types (GGUF/llama token_type values).
    enum Type : int32_t {
        TYPE_UNDEFINED = 0, TYPE_NORMAL = 1, TYPE_UNKNOWN = 2,
        TYPE_CONTROL = 3, TYPE_USER_DEFINED = 4, TYPE_UNUSED = 5, TYPE_BYTE = 6,
    };
    int32_t token_type(int32_t t) const { return types_[t]; }

private:
    void tokenize_fragment(const std::string& text, std::vector<int32_t>& out) const;

    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<int32_t> types_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    int32_t byte_to_id_[256];
    std::vector<int32_t> specials_;  // ids, sorted by text length desc
    int32_t bos_ = -1;
    int32_t eos_ = -1;
    bool add_space_prefix_ = true;   // SPM default (llama.cpp)
    bool escape_whitespaces_ = true; // SPM default (llama.cpp)
};

} // namespace haidass
