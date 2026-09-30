// tokenizer.cpp — llama.cpp-compatible SPM-session tokenizer.
//
// Reference (algorithm and tie-breaking mirrored from):
//   llama.cpp src/llama-vocab.cpp  (llm_tokenizer_spm_session,
//   tokenizer_st_partition, llama_escape_whitespace, token_to_piece)

#include "tokenizer.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <queue>

namespace haidass {

namespace {

// UTF-8 sequence length from the leading byte (llama.cpp unicode_len_utf8).
inline size_t utf8_len(char c) {
    static const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return lookup[static_cast<uint8_t>(c) >> 4];
}

constexpr const char* kSpaceMark = "\xe2\x96\x81";  // U+2581

std::string escape_whitespace(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] == ' ') {
            out += kSpaceMark;
        } else {
            out += text[i];
        }
    }
    return out;
}

std::string unescape_whitespace(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (i + 2 < text.size() + 1 && text.compare(i, 3, kSpaceMark) == 0) {
            out += ' ';
            i += 3;
        } else {
            out += text[i];
            ++i;
        }
    }
    return out;
}

bool parse_byte_token(const std::string& s, int& byte) {
    // "<0xHH>" — 6 chars
    if (s.size() != 6 || s[0] != '<' || s[1] != '0' || (s[2] != 'x' && s[2] != 'X') || s[5] != '>') {
        return false;
    }
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    int hi = hex(s[3]), lo = hex(s[4]);
    if (hi < 0 || lo < 0) return false;
    byte = hi * 16 + lo;
    return true;
}

// --- SPM session symbols ---------------------------------------------------

struct Symbol {
    int prev;
    int next;
    const char* text;
    size_t n;
};

struct Bigram {
    int left;
    int right;
    float score;
    size_t size;
    // priority queue top = highest score; tie -> smallest left index
    struct Cmp {
        bool operator()(const Bigram& l, const Bigram& r) const {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
};

} // namespace

bool Tokenizer::load(const GgufFile& gguf, std::string* err) {
    if (gguf.get_str("tokenizer.ggml.model") != "llama") {
        if (err) *err = "unsupported tokenizer model: " + gguf.get_str("tokenizer.ggml.model");
        return false;
    }
    const GgufValue* toks = gguf.find("tokenizer.ggml.tokens");
    const GgufValue* scores = gguf.find("tokenizer.ggml.scores");
    const GgufValue* types = gguf.find("tokenizer.ggml.token_type");
    if (!toks || toks->type != GV_ARRAY || toks->elem_type != GV_STRING) {
        if (err) *err = "missing tokenizer.ggml.tokens";
        return false;
    }
    const size_t n = toks->strs.size();
    if (!scores || scores->nums.size() != n || !types || types->nums.size() != n) {
        if (err) *err = "tokenizer scores/types missing or size mismatch";
        return false;
    }
    tokens_ = toks->strs;
    scores_.resize(n);
    types_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        scores_[i] = static_cast<float>(scores->nums[i]);
        types_[i] = static_cast<int32_t>(types->nums[i]);
        token_to_id_[tokens_[i]] = static_cast<int32_t>(i);
    }

    std::fill(std::begin(byte_to_id_), std::end(byte_to_id_), -1);
    specials_.clear();
    for (size_t i = 0; i < n; ++i) {
        if (types_[i] == TYPE_BYTE) {
            int b = 0;
            if (parse_byte_token(tokens_[i], b)) byte_to_id_[b] = static_cast<int32_t>(i);
        }
        if (types_[i] == TYPE_CONTROL || types_[i] == TYPE_USER_DEFINED || types_[i] == TYPE_UNKNOWN) {
            specials_.push_back(static_cast<int32_t>(i));
        }
    }
    std::sort(specials_.begin(), specials_.end(),
              [&](int32_t a, int32_t b) { return tokens_[a].size() > tokens_[b].size(); });

    bos_ = static_cast<int32_t>(gguf.get_u32("tokenizer.ggml.bos_token_id", UINT32_MAX));
    eos_ = static_cast<int32_t>(gguf.get_u32("tokenizer.ggml.eos_token_id", UINT32_MAX));
    if (bos_ == (int32_t)UINT32_MAX) bos_ = -1;
    if (eos_ == (int32_t)UINT32_MAX) eos_ = -1;

    // llama.cpp SPM defaults; the GGUF kv keys may override them.
    add_space_prefix_ = gguf.get_bool("tokenizer.ggml.add_space_prefix", true);
    escape_whitespaces_ = true;
    return true;
}

// Greedy bigram-merge session over one text fragment (llm_tokenizer_spm_session).
void Tokenizer::tokenize_fragment(const std::string& text, std::vector<int32_t>& out) const {
    if (text.empty()) return;

    std::vector<Symbol> symbols;
    symbols.reserve(text.size());
    size_t offs = 0;
    int index = 0;
    while (offs < text.size()) {
        Symbol sym;
        size_t len = utf8_len(text[offs]);
        sym.text = text.data() + offs;
        sym.n = std::min(len, text.size() - offs);
        offs += sym.n;
        sym.prev = index - 1;
        sym.next = offs == text.size() ? -1 : index + 1;
        symbols.push_back(sym);
        ++index;
    }

    std::priority_queue<Bigram, std::vector<Bigram>, Bigram::Cmp> work_queue;
    std::unordered_map<std::string, std::pair<int, int>> rev_merge;

    auto try_add_bigram = [&](int left, int right) {
        if (left == -1 || right == -1) return;
        std::string cat(symbols[left].text, symbols[left].n);
        cat.append(symbols[right].text, symbols[right].n);
        auto it = token_to_id_.find(cat);
        if (it == token_to_id_.end()) return;
        Bigram b;
        b.left = left;
        b.right = right;
        b.score = scores_[it->second];
        b.size = cat.size();
        work_queue.push(b);
        rev_merge[std::move(cat)] = {left, right};
    };

    for (int i = 1; i < (int)symbols.size(); ++i) try_add_bigram(i - 1, i);

    while (!work_queue.empty()) {
        Bigram bigram = work_queue.top();
        work_queue.pop();
        Symbol& left_sym = symbols[bigram.left];
        Symbol& right_sym = symbols[bigram.right];
        if (left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size) {
            continue;
        }
        left_sym.n += right_sym.n;
        right_sym.n = 0;
        left_sym.next = right_sym.next;
        if (right_sym.next >= 0) symbols[right_sym.next].prev = bigram.left;
        try_add_bigram(left_sym.prev, bigram.left);
        try_add_bigram(bigram.left, left_sym.next);
    }

    // Emit final symbols; recursively split merged spans not in the vocab,
    // and byte-fallback anything still unknown.
    std::function<void(const Symbol&)> resegment = [&](const Symbol& sym) {
        std::string text(sym.text, sym.n);
        auto it = token_to_id_.find(text);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
            return;
        }
        auto m = rev_merge.find(text);
        if (m == rev_merge.end()) {
            for (size_t j = 0; j < sym.n; ++j) {
                int32_t id = byte_to_id_[static_cast<uint8_t>(sym.text[j])];
                if (id >= 0) out.push_back(id);
            }
            return;
        }
        resegment(symbols[m->second.first]);
        resegment(symbols[m->second.second]);
    };

    for (int i = 0; i != -1; i = symbols[i].next) resegment(symbols[i]);
}

std::vector<int32_t> Tokenizer::encode(const std::string& raw_text, bool parse_special) const {
    std::vector<int32_t> out;
    if (raw_text.empty()) return out;

    // Fragment list: (is_special, token_id) or raw text span.
    struct Fragment {
        bool is_special;
        int32_t token;
        size_t offset, length;
    };
    std::vector<Fragment> frags{{false, -1, 0, raw_text.size()}};

    if (parse_special) {
        for (int32_t sid : specials_) {
            const std::string& stext = tokens_[sid];
            if (stext.empty()) continue;
            std::vector<Fragment> next;
            for (const Fragment& f : frags) {
                if (f.is_special) {
                    next.push_back(f);
                    continue;
                }
                size_t base = f.offset;
                size_t end = f.offset + f.length;
                while (true) {
                    size_t match = raw_text.find(stext, base);
                    if (match == std::string::npos || match + stext.size() > end) break;
                    if (match > base) next.push_back({false, -1, base, match - base});
                    next.push_back({true, sid, match, stext.size()});
                    base = match + stext.size();
                }
                if (base < end) next.push_back({false, -1, base, end - base});
            }
            frags.swap(next);
        }
    }

    bool is_prev_special = true;  // prefix first text fragment with a space
    for (const Fragment& f : frags) {
        if (f.is_special) {
            out.push_back(f.token);
            is_prev_special = true;
            continue;
        }
        std::string text;
        if (add_space_prefix_ && is_prev_special) text = ' ';
        text.append(raw_text, f.offset, f.length);
        if (escape_whitespaces_) text = escape_whitespace(text);
        tokenize_fragment(text, out);
        is_prev_special = false;
    }
    return out;
}

std::string Tokenizer::piece(int32_t token, bool special) const {
    if (token < 0 || token >= (int32_t)tokens_.size()) return "";
    const int32_t t = types_[token];
    if (!special && (t == TYPE_UNKNOWN || t == TYPE_CONTROL)) return "";
    if (t == TYPE_CONTROL || t == TYPE_UNKNOWN || t == TYPE_USER_DEFINED) {
        return tokens_[token];
    }
    if (t == TYPE_BYTE) {
        int b = 0;
        if (parse_byte_token(tokens_[token], b)) return std::string(1, static_cast<char>(b));
        return "";
    }
    // NORMAL / UNUSED / UNDEFINED
    if (escape_whitespaces_) return unescape_whitespace(tokens_[token]);
    return tokens_[token];
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens, bool special) const {
    std::string out;
    for (int32_t t : tokens) out += piece(t, special);
    return out;
}

} // namespace haidass
