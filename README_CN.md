# haidass-translate-cpp

![haidass-translate-cpp — High-performance C++ translation engine](docs/main.png)

纯 C++17 实现的 [Haidass-Translate-143M](https://huggingface.co/DALabCommunity/Haidass-Translate-143M) 中英互译推理引擎。**零第三方 C++ 依赖**（仅标准库 + 线程），支持两种交付模式：

- **文件加载模式**：运行时通过 `-m model.gguf` 加载 GGUF 权重（mmap，几乎零内存拷贝）。
- **内嵌极致模式**：权重在编译期直接打进二进制，产出一个**完全自包含的单文件可执行程序**，免任何外部文件。

[English](README.md)

## 特性

- Qwen3 架构（30 层 / hidden 576 / GQA 9Q-3KV / QK-Norm / RoPE），全 F32 计算（官方报告 fp16 激活会溢出）
- 直接读取官方 GGUF（推荐 [umeiko/Haidass-Translate-143M-GGUF](https://huggingface.co/umeiko/Haidass-Translate-143M-GGUF) 的 Q8_0，188 MB）
- 内置分词器：完整复刻 llama.cpp 的 SPM 会话语义（特殊 token 预切分、每段 ▁ 前缀、score 堆归并、`<0xXX>` 字节回退），与训练时逐 token 一致
- SIMD：x86-64 AVX2/FMA 与 ARM NEON 独立编译单元 + 运行时分发，纯标量回退可用于任何平台（含 RISC-V）
- 多线程 batch GEMM（Q8_0 反量化 × F32）
- 端到端输出与 HF transformers（加载同一 Q8_0）**逐字一致**，logits cosine = 1.000000

## 快速开始

### 下载权重

```bash
python tools/download_model.py        # 默认下载 Q8_0 到 models/
# 或手动: https://huggingface.co/umeiko/Haidass-Translate-143M-GGUF
```

### 构建（文件加载模式）

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Windows MSVC：去掉 `-G Ninja` 用默认 Visual Studio 生成器，然后
`cmake --build build --config Release`。Windows 本机逐步编译教程
（PowerShell/cmd 两种写法，用 VS BuildTools 自带的 CMake+Ninja，零额外安装）见
[docs/BUILD_WINDOWS.md](docs/BUILD_WINDOWS.md)。

MinGW/MSYS2 无 CMake 的极简回退：`bash scripts/build_mingw.sh`。
Windows MinGW 逐步教程（编译器去哪下、MSYS2 与 winlibs 二选一）见
[docs/BUILD_WINDOWS_MINGW.md](docs/BUILD_WINDOWS_MINGW.md)。

### 构建（内嵌单文件模式）

```bash
cmake -S . -B build-embed -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DHAIDASS_BUILD_TESTS=OFF \
  -DHAIDASS_EMBED_MODEL=/path/to/haidass-translate-143m-q8_0.gguf
cmake --build build-embed -j
# 产出单个 haidass 可执行文件（≈188 MB），权重已在其中
```

GCC/Clang/MinGW 用 `.incbin` 内嵌，MSVC 用 Windows `.rc` 资源；同一份 GGUF 解析代码，运行期零解析开销、直接 mmap 只读段。

### 使用

```bash
./build/haidass -m models/haidass-translate-143m-q8_0.gguf "Machine translation bridges languages and cultures."
# => 机器翻译跨越语言和文化。

./build/haidass -m models/haidass-translate-143m-q8_0.gguf "机器翻译连接了不同的语言与文化。"
# => Machine translation connects different languages and cultures.

echo "Hello" | ./build/haidass -m model.gguf     # stdin 也可以
./build-embed/haidass "你好世界"                  # 内嵌版免 -m
```

方向自动检测（含 CJK 即中→英），也可 `--en2zh` / `--zh2en` 强制。默认贪心解码；`--temp/--top-k/--top-p/--seed` 开启采样。完整参数见 `--help`。

## 交叉编译

提供现成 toolchain 文件（全静态链接）：

```bash
cmake -S . -B build-arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64-linux-gnu.cmake
cmake -S . -B build-armhf -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-linux-gnueabihf.cmake
cmake -S . -B build-riscv -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/riscv64-linux-gnu.cmake
cmake -S . -B build-win   -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/x86_64-w64-mingw32.cmake
```

CI 覆盖：Linux x86_64 / aarch64 / armhf / riscv64、macOS x86_64 / arm64、Windows MSVC(x64/x86) / MinGW，交叉目标在 qemu-user 下跑完整测试。打 `v*` tag 触发 release 工作流，自动发布各平台的文件版与内嵌版 zip。

## 测试

```bash
ctest --test-dir build --output-on-failure
```

7 项测试：GGUF 解析、分词器（对拍 llama.cpp 语义参考实现）、算子、合成 fixture 模型前向 logits 对拍 numpy 参考、端到端生成。另外 `tools/verify_vs_hf.py` 用 HF transformers 加载同一 GGUF 做整模型对照（需 `pip install -r tools/requirements.txt` + torch/transformers）。

## 性能参考

Q8_0 模型，贪心解码：约 **40 tok/s**（x86-64 AVX2，24 线程桌面）。143M 小模型在 ARM/RISC-V 开发板上也可实用。

## 架构

```
src/
  platform.{h,cpp}     mmap/文件、UTF-8 控制台、线程数等 OS 抽象
  gguf.{h,cpp}         GGUF v3 解析（两种模式共用同一份代码）
  tokenizer.{h,cpp}    llama.cpp SPM 会话分词器
  kernels.h            内核接口（GEMM 等）
  kernels_scalar.cpp   纯标量参考内核（任何平台可跑）
  kernels_avx2.cpp     AVX2/FMA（独立 TU，-mavx2 -mfma）
  kernels_neon.cpp     NEON（独立 TU）
  cpu_dispatch.cpp     运行时分发（cpuid / HWCAP）
  thread_pool.{h,cpp}  极简线程池
  model.{h,cpp}        Qwen3 前向 + KV cache + 权重视图
  ops.{h,cpp}          rmsnorm/rope/attention/silu 等
  sampler.{h,cpp}      贪心 / top-k / top-p 采样
  generate.{h,cpp}     生成循环（chat 模板、EOS 处理）
  main.cpp             CLI
  embed_{none,asm,res}.cpp  三种内嵌后端（无内嵌/.incbin/.rc）
tools/                 Python 工具：下载、fixture/测试向量生成、HF 对照
tests/                 C++ 测试 + 提交的 vocab.bin（从官方 GGUF 抽取）
cmake/toolchains/      交叉编译 toolchain 文件
```

## 致谢

- 模型：[DALabCommunity/Haidass-Translate-143M](https://huggingface.co/DALabCommunity/Haidass-Translate-143M)
- GGUF 量化：[umeiko/Haidass-Translate-143M-GGUF](https://huggingface.co/umeiko/Haidass-Translate-143M-GGUF)
- 分词器语义参考 [llama.cpp](https://github.com/ggml-org/llama.cpp)
