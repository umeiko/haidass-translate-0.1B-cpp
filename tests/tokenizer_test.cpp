// tokenizer_test.cpp — C++ tokenizer vs reference vectors (base64\tids lines).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "model.h"
#include "platform.h"
#include "test_util.h"

using namespace haidass;

namespace {

std::string b64_decode(const std::string& in) {
    static const char* chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, bits = -8;
    for (char c : in) {
        const char* p = std::strchr(chars, c);
        if (!p) break;
        val = (val << 6) + int(p - chars);
        bits += 6;
        if (bits >= 0) {
            out.push_back(char((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: tokenizer_test <fixture.gguf> <vectors.txt>\n");
        return 2;
    }
    MappedFile mf;
    std::string err;
    CHECK(mf.open(argv[1], &err));
    if (!mf.is_open()) return test_summary("tokenizer_test");
    Model model;
    CHECK(model.open(mf.data(), mf.size(), &err));
    if (g_failures) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return test_summary("tokenizer_test");
    }
    const Tokenizer& tok = model.tokenizer();
    CHECK(tok.n_tokens() == 64000);
    CHECK(tok.eos() == 5);

    std::ifstream in(argv[2]);
    CHECK(in.good());
    std::string line;
    int cases = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const size_t tab = line.find('\t');
        CHECK(tab != std::string::npos);
        if (tab == std::string::npos) continue;
        const std::string text = b64_decode(line.substr(0, tab));
        std::vector<int32_t> want;
        {
            std::stringstream ss(line.substr(tab + 1));
            std::string id;
            while (std::getline(ss, id, ',')) want.push_back(std::stoi(id));
        }
        std::vector<int32_t> got = tok.encode(text, true);
        ++cases;
        if (got != want) {
            std::fprintf(stderr, "FAIL encode mismatch for text #%d (bytes %zu)\n  want:",
                         cases, text.size());
            for (int32_t t : want) std::fprintf(stderr, " %d", t);
            std::fprintf(stderr, "\n  got: ");
            for (int32_t t : got) std::fprintf(stderr, " %d", t);
            std::fprintf(stderr, "\n");
            ++g_failures;
        }
        // round-trip: decode(encode(x)) for texts without specials
        if (text.find("<|") == std::string::npos) {
            const std::string dec = tok.decode(want);
            CHECK(dec == tok.decode(got));
        }
    }
    CHECK(cases >= 15);

    // decode sanity: special token pieces
    CHECK(tok.piece(5, /*special=*/false).empty());      // <|im_end|> control: skipped
    CHECK(tok.piece(5, /*special=*/true) == "<|im_end|>");

    return test_summary("tokenizer_test");
}
