// main.cpp — haidass CLI: bidirectional zh<->en translation.
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include "generate.h"
#include "kernels.h"
#include "model.h"
#include "platform.h"

namespace {

const char* kUsage = R"(haidass — Haidass-Translate-143M 中英互译 (pure C++, zero deps)

Usage:
  haidass [options] "text"          translate text (direction auto-detected)
  echo "text" | haidass [options]   read text from stdin

Options:
  -m, --model PATH       GGUF model file (omit in embedded builds)
      --en2zh            force English -> Chinese
      --zh2en            force Chinese -> English
  -n, --max-tokens N     max generated tokens (default 256)
  -c, --ctx N            context size (default 2048)
  -t, --threads N        worker threads (default: hardware concurrency)
      --temp T           sampling temperature, 0 = greedy (default 0)
      --top-k K          top-k (default 40, sampling only)
      --top-p P          top-p (default 0.9, sampling only)
      --seed S           RNG seed (default 42)
      --embed-info       print embedded model info and exit
  -h, --help             this help
)";

bool has_cjk(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t c = (uint8_t)s[i];
        uint32_t cp;
        size_t len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { ++i; continue; }
        if (i + len > s.size()) break;
        for (size_t j = 1; j < len; ++j) cp = (cp << 6) | ((uint8_t)s[i + j] & 0x3F);
        i += len;
        if ((cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
            (cp >= 0x20000 && cp <= 0x2A6DF)) {
            return true;
        }
    }
    return false;
}

std::string build_prompt(const std::string& text, bool en2zh) {
    const char* instr = en2zh ? "Translate the following text from English to Simplified Chinese."
                              : "请将以下简体中文翻译成英文。";
    return std::string("<|im_start|>user\n") + instr + "\n" + text +
           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

} // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    haidass::kernels_init();

    // On Windows, CRT main() mangles non-ASCII argv into the ANSI codepage.
    std::vector<std::string> args;
    {
        std::vector<std::string> u8 = haidass::platform_utf8_args();
        if (!u8.empty()) {
            args.assign(u8.begin() + 1, u8.end());  // skip exe name
        } else {
            for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
        }
    }

    std::string model_path;
    std::string text;
    haidass::GenParams params;
    int direction = 0;  // 0=auto 1=en2zh 2=zh2en
    int threads = (int)std::thread::hardware_concurrency();
    bool embed_info = false;

    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return args[++i];
        };
        if (a == "-m" || a == "--model") model_path = next("--model");
        else if (a == "--en2zh") direction = 1;
        else if (a == "--zh2en") direction = 2;
        else if (a == "-n" || a == "--max-tokens") params.max_tokens = std::stoi(next("--max-tokens"));
        else if (a == "-c" || a == "--ctx") params.max_ctx = std::stoi(next("--ctx"));
        else if (a == "-t" || a == "--threads") threads = std::stoi(next("--threads"));
        else if (a == "--temp") params.temp = std::stof(next("--temp"));
        else if (a == "--top-k") params.top_k = std::stoi(next("--top-k"));
        else if (a == "--top-p") params.top_p = std::stof(next("--top-p"));
        else if (a == "--seed") params.seed = std::stoull(next("--seed"));
        else if (a == "--embed-info") embed_info = true;
        else if (a == "-h" || a == "--help") { std::fputs(kUsage, stdout); return 0; }
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); return 1; }
        else { if (!text.empty()) text += " "; text += a; }
    }

    const uint8_t* model_data = haidass::embedded_model_data();
    size_t model_size = haidass::embedded_model_size();
    haidass::MappedFile mapped;

    if (model_data) {
        if (model_path.size()) {
            std::fprintf(stderr, "note: -m ignored, this binary has an embedded model\n");
        }
    } else {
        if (embed_info) { std::puts("no embedded model (file-mode build)"); return 0; }
        if (model_path.empty()) {
            std::fprintf(stderr, "error: no model file given (-m). See --help.\n");
            return 1;
        }
        std::string err;
        if (!mapped.open(model_path, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        model_data = mapped.data();
        model_size = mapped.size();
    }

    haidass::Model model;
    {
        std::string err;
        if (!model.open(model_data, model_size, &err)) {
            std::fprintf(stderr, "error: failed to load model: %s\n", err.c_str());
            return 1;
        }
    }

    if (embed_info) {
        std::printf("embedded model: %zu bytes, %s, kernel tier: %s\n", model_size,
                    model.weight_summary().c_str(), haidass::kernels_active_tier());
        return 0;
    }

    if (text.empty()) {
        std::string line, all;
        while (std::getline(std::cin, line)) { all += line; all += '\n'; }
        if (!all.empty() && all.back() == '\n') all.pop_back();
        text = all;
    }
    if (text.empty()) {
        std::fprintf(stderr, "error: no input text. See --help.\n");
        return 1;
    }

    const bool en2zh = direction == 0 ? !has_cjk(text) : direction == 1;
    const std::string prompt = build_prompt(text, en2zh);

    std::fprintf(stderr, "[%s | %s | threads=%d]\n", en2zh ? "en→zh" : "zh→en",
                 haidass::kernels_active_tier(), threads);

    haidass::Generator gen(model, threads);
    haidass::GenStats stats;
    std::string err;
    bool ok = gen.generate(prompt, params,
                           [](const std::string& piece) {
                               std::fwrite(piece.data(), 1, piece.size(), stdout);
                               std::fflush(stdout);
                               return true;
                           },
                           &stats, &err);
    std::fputc('\n', stdout);
    if (!ok) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "[prompt %d tok (%.1f tok/s) | gen %d tok (%.1f tok/s)]\n",
                 stats.prompt_tokens,
                 stats.prefill_s > 0 ? stats.prompt_tokens / stats.prefill_s : 0.0,
                 stats.gen_tokens,
                 stats.decode_s > 0 ? stats.gen_tokens / stats.decode_s : 0.0);
    return 0;
}
