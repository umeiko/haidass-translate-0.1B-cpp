// e2e_fixture_test.cpp — generation loop smoke test on the fixture model:
// deterministic greedy output across runs and thread counts, stops properly.
#include <cstdint>
#include <string>

#include "generate.h"
#include "kernels.h"
#include "model.h"
#include "platform.h"
#include "test_util.h"

using namespace haidass;

namespace {

struct Result {
    std::string text;
    GenStats stats;
    bool ok = false;
};

Result run(const Model& model, const std::string& prompt, int threads, int max_tokens) {
    Generator gen(model, threads);
    GenParams params;
    params.max_tokens = max_tokens;
    params.max_ctx = 512;
    params.temp = 0.0f;
    Result r;
    std::string err;
    r.ok = gen.generate(prompt, params,
                        [&](const std::string& piece) { r.text += piece; return true; },
                        &r.stats, &err);
    if (!r.ok) std::fprintf(stderr, "generate failed: %s\n", err.c_str());
    return r;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: e2e_fixture_test <fixture.gguf>\n");
        return 2;
    }
    kernels_init();

    MappedFile mf;
    std::string err;
    CHECK(mf.open(argv[1], &err));
    if (!mf.is_open()) return test_summary("e2e_fixture_test");

    Model model;
    CHECK(model.open(mf.data(), mf.size(), &err));
    if (g_failures) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return test_summary("e2e_fixture_test");
    }

    const std::string prompt =
        "<|im_start|>user\nTranslate the following text from English to Simplified Chinese.\n"
        "Hello world.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    Result r1 = run(model, prompt, 1, 32);
    Result r2 = run(model, prompt, 1, 32);
    Result r4 = run(model, prompt, 4, 32);

    CHECK(r1.ok && r2.ok && r4.ok);
    CHECK(r1.stats.gen_tokens > 0);
    CHECK(r1.stats.gen_tokens <= 32);
    CHECK(r1.text == r2.text);                     // deterministic
    CHECK(r1.text == r4.text);                     // thread-count independent
    CHECK(r1.stats.prompt_tokens == r4.stats.prompt_tokens);

    std::fprintf(stderr, "gen %d tokens, prompt %d tokens, %zu bytes out\n",
                 r1.stats.gen_tokens, r1.stats.prompt_tokens, r1.text.size());

    return test_summary("e2e_fixture_test");
}
