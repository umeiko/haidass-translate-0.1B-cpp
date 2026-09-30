# Windows 本地编译指南（MinGW 路线）

本文档面向 Windows 本机用 **MinGW-w64 GCC** 编译 `haidass.exe`。这是本仓库作者在 AGENTS.md 里记录的本地开发路线；CI 的 `windows-mingw` job 做的是 Linux→Windows 交叉编译，与本机路线不同。MSVC 路线见 [BUILD_WINDOWS.md](BUILD_WINDOWS.md)。

整体思路（与 AGENTS.md 一致）：

- **编译器 gcc/g++** 来自 MSYS2 的 MINGW64 环境（或独立压缩包）
- **cmake / ninja / numpy** 来自项目里的 Python venv——不要用系统 Python（没装 numpy），也不依赖 MSYS2 的 cmake

## 第 1 步：获取 MinGW-w64 编译器（二选一）

### 选项 A：MSYS2（推荐，可用 pacman 持续更新）

1. 下载安装器：<https://www.msys2.org/>（或 `winget install MSYS2.MSYS2`），默认装到 `C:\msys64`。
2. 从开始菜单打开 **"MSYS2 MINGW64"**（注意是 MINGW64，不是 UCRT64 也不是 MSYS）。
3. 先更新包管理器（可能要关掉窗口重开一次再执行第二条）：

   ```bash
   pacman -Syu
   pacman -Su
   ```

4. 安装 64 位 GCC 工具链：

   ```bash
   pacman -S mingw-w64-x86_64-toolchain
   ```

   装完编译器就在 `C:\msys64\mingw64\bin\g++.exe`。

   > 如果更想要 UCRT 运行库（UCRT64 环境），装 `mingw-w64-ucrt-x86_64-toolchain`，
   > 后文所有 `/c/msys64/mingw64` 路径改成 `/c/msys64/ucrt64` 即可。AGENTS.md 记录的是 MINGW64。

### 选项 B：winlibs 独立压缩包（免安装，不想要 MSYS2 时用）

1. 打开 <https://winlibs.com/>，下载 **GCC (MinGW-w64) Win64 UCRT runtime** 的 7z/zip 压缩包。
2. 解压到例如 `C:\mingw64`，编译器在 `C:\mingw64\bin\g++.exe`。
3. 后文所有 `/c/msys64/mingw64/bin` 路径替换成 `/c/mingw64/bin`。

## 第 2 步：准备 Python venv（cmake / ninja / numpy 都从这里来）

在项目根目录（cmd 或 PowerShell 均可）：

```cmd
python -m venv .venv
.venv\Scripts\python -m pip install -r tools\requirements.txt cmake ninja
```

`tools\requirements.txt` 提供 numpy（跑测试生成向量用）；`cmake`、`ninja` 是 pip 分发的 Windows 原生二进制，装完位于 `.venv\Scripts\cmake.exe` / `ninja.exe`。

## 第 3 步：编译 + 测试

打开 **Git Bash**（或 MSYS2 MINGW64 shell），`cd` 到项目根目录，然后：

```bash
export PATH="/c/msys64/mingw64/bin:$PWD/.venv/Scripts:$PATH"
./.venv/Scripts/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DPython3_EXECUTABLE="$PWD/.venv/Scripts/python.exe"
./.venv/Scripts/cmake --build build -j
./.venv/Scripts/ctest --test-dir build --output-on-failure
```

要点：

- `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++`：强制选 MinGW GCC，防止 cmake 捡到 MSVC。
- `-DPython3_EXECUTABLE=...venv...`：测试向量生成必须用 venv 的 Python（系统 Python 没 numpy）。
- 产物：**`build/haidass.exe`**（MinGW 静态链接，单文件可直接拷走）。

## 第 4 步：下载模型并运行

与 MSVC 路线相同（详见 [BUILD_WINDOWS.md](BUILD_WINDOWS.md#下载模型并运行)）：

```bash
python tools/download_model.py
./build/haidass.exe -m models/haidass-translate-143m-q8_0.gguf "Hello world"
```

## 极简回退：不用 CMake

`scripts/build_mingw.sh` 是纯 g++ 的逐文件编译脚本（无 cmake/ninja/venv 依赖，但需要 bash 环境）。在 **MSYS2 MINGW64** 或 Git Bash 里：

```bash
export PATH="/c/msys64/mingw64/bin:$PATH"
bash scripts/build_mingw.sh        # 产物 out-mingw/haidass.exe
```

注意它只编译 CLI 本体，不构建测试。

## 常见问题

- **shell 选错**：MSYS2 开始菜单有好几个入口，务必选 **MINGW64**（或对应 UCRT64），选成 "MSYS2 MSYS" 装出来的 gcc 是 cygwin 式的 POSIX 层，不是原生 Windows 编译器。
- **cmake 捡错编译器**：如果 `build/` 目录之前用 MSVC 配置过，删掉 `build/` 重来；CMake 缓存不会换编译器。
- **测试在 gen_vectors 失败**：基本是 Python 没用 venv 的（缺 numpy），检查第 3 步的 `-DPython3_EXECUTABLE`。
- **huggingface.co 连不上**：`tools/download_model.py` 默认就走 hf-mirror.com 镜像，断了重跑会断点续传。
