# AGENTS.md

Haidass-Translate-143M 的纯 C++17 推理引擎。零第三方 C++ 依赖，两种模式：文件加载（mmap GGUF）与内嵌单文件（`.incbin` / `.rc`）。

## 构建与测试

```bash
# 本地（Windows + MSYS2 MinGW）：必须用 venv 里的 cmake/python（系统 Python 无 numpy）
export PATH="/c/msys64/mingw64/bin:$PWD/.venv/Scripts:$PATH"
./.venv/Scripts/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DPython3_EXECUTABLE="$PWD/.venv/Scripts/python.exe"
./.venv/Scripts/cmake --build build -j
./.venv/Scripts/ctest --test-dir build --output-on-failure
```

- **本地只验证 MinGW**；aarch64 / armhf / riscv64 / MSVC / macOS 一律交 CI（`.github/workflows/ci.yml`，交叉目标在 qemu-user 下跑测试）。不要在本机尝试交叉编译验证。
- 本机逐步编译教程：`docs/BUILD_WINDOWS_MINGW.md`（MinGW，含编译器下载）、`docs/BUILD_WINDOWS.md`（MSVC）。
- Python 依赖只给工具/测试用：`pip install -r tools/requirements.txt`（venv 在 `.venv/`）。
- 真模型端到端对照：`PYTHONIOENCODING=utf-8 python tools/verify_vs_hf.py --logits-bin build/hf_logits.bin`，再跑 `build/forward_logits_test.exe <gguf> build/hf_logits.bin`。

## 硬性约定

- C++17，禁止引入第三方 C++ 库。
- 全 F32 计算（官方报告 fp16 激活溢出）。线性层权重 Q8_0，embedding BF16（按需转 F32）。
- SIMD 内核必须放独立编译单元并加 ISA 编译选项，运行时分发（`cpu_dispatch.cpp`）；标量路径必须始终可用（RISC-V 只靠它）。
- 分词器语义 = llama.cpp `llm_tokenizer_spm_session`：特殊 token（type 2/3/4）预切分、**每个** raw 段加 ▁ 前缀、`\n` 显式编码为 `<0x0A>`、score 大顶堆归并、字节回退 `<0xXX>`。**不是**原生 sentencepiece 行为（原生会吞 `\n`，模型收到会直接输出 `<|im_end|>`）。改分词器必须同步改 `tools/gen_vectors.py` 的 `ReferenceTokenizer` 并重生成测试向量。
- GGUF 张量名是 llama.cpp 风格：`blk.N.attn_output.weight`（不是 `attn_o`）。embedding 层名 `token_embd.weight`，tie embeddings（无 lm_head）。
- 修改 `src/` 结构、构建选项、测试流程时，同步更新本文件与 README.md。

## 关键文件

- `CMakeLists.txt`：选项 `HAIDASS_BUILD_TESTS` / `HAIDASS_SIMD` / `HAIDASS_EMBED_MODEL` / `HAIDASS_BUILD_CLI`
- `cmake/toolchains/*.cmake`：aarch64 / armhf / riscv64 / mingw 交叉链（全静态）
- `tests/data/vocab.bin`：已提交的 64000 词表（`tools/extract_vocab.py` 从官方 GGUF 抽取），测试向量由此生成
- `.github/workflows/release.yml`：tag `v*` 触发，发布各平台文件版 + Q8_0 内嵌版

## 环境备忘

- 官方 GGUF：`models/haidass-translate-143m-q8_0.gguf`（188214496 字节，不提交 git）
- huggingface.co 直连不稳时用 `hf-mirror.com`；PyPI 装不上时用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- Windows 下 CLI 的 argv 经 `platform_utf8_args()`（`GetCommandLineW`）重建，勿删
