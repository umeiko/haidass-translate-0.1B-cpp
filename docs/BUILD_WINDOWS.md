# Windows 本地编译指南

本文档面向 Windows 本机从零编译 `haidass.exe`。推荐路线是 **MSVC（VS 2022 BuildTools 自带 CMake/Ninja，无需单独安装）**，与 CI 的 `windows-msvc` job 一致。MSYS2/MinGW 路线见根目录 `AGENTS.md`。

## 前置条件

| 组件 | 检查方法 | 说明 |
|---|---|---|
| VS 2022 BuildTools（含 C++ 工作负载） | 存在目录 `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` | 只需 BuildTools，不用完整 Visual Studio |
| Python 3 | `python --version` | 仅用于下载模型 / 生成测试向量，编译本身不需要 |

CMake 和 Ninja **不用单独安装**，使用 BuildTools 自带的：

```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\
├── CMake\bin\cmake.exe
└── Ninja\ninja.exe
```

## 方式一：PowerShell（推荐）

在**项目根目录**打开 PowerShell，依次执行：

```powershell
# 1. 加载 MSVC x64 编译环境（让 cl.exe 进 PATH）
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation

# 2. 设置 cmake / ninja 路径
$VSBT = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake"
$env:PATH = "$VSBT\Ninja;$env:PATH"

# 3. 配置 + 编译（PowerShell 中调用变量里的 exe 要加 &）
& "$VSBT\CMake\bin\cmake.exe" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAIDASS_BUILD_TESTS=OFF
& "$VSBT\CMake\bin\cmake.exe" --build build -j

# 4. 验证
.\build\haidass.exe --help
```

产物：**`build\haidass.exe`**。

> 第 1 步如果报执行策略错误（红字提到 execution policy），先执行
> `Set-ExecutionPolicy -Scope Process Bypass` 再重跑第 1 步。

## 方式二：x64 Native Tools Command Prompt（cmd）

开始菜单打开 **"x64 Native Tools Command Prompt for VS 2022"**（只有这个窗口里 `cl.exe` 才在 PATH 里），然后：

```cmd
cd /d D:\path\to\haidass-translate-0.1B-cpp

set "VSBT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake"
set "PATH=%VSBT%\Ninja;%PATH%"

"%VSBT%\CMake\bin\cmake.exe" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAIDASS_BUILD_TESTS=OFF
"%VSBT%\CMake\bin\cmake.exe" --build build -j

build\haidass.exe --help
```

> 注意：`set "VAR=..."` 和 `%VAR%` 是 cmd 语法，**在 PowerShell 里无效**——PowerShell 请用方式一。

## 下载模型并运行

模型（Q8_0，约 188 MB）不随 git 分发，用自带脚本下载（默认走 hf-mirror 国内镜像，支持断点续传）：

```powershell
python tools\download_model.py
# 下载到 models\haidass-translate-143m-q8_0.gguf
```

运行：

```powershell
.\build\haidass.exe -m models\haidass-translate-143m-q8_0.gguf "Hello world"
.\build\haidass.exe -m models\haidass-translate-143m-q8_0.gguf --zh2en "你好，世界"
```

## 可选：跑测试套件

测试需要 Python + numpy 生成测试向量：

```powershell
pip install numpy
& "$VSBT\CMake\bin\cmake.exe" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
& "$VSBT\CMake\bin\cmake.exe" --build build -j
& "$VSBT\CMake\bin\ctest.exe" --test-dir build --output-on-failure
```

（重新配置时去掉 `-DHAIDASS_BUILD_TESTS=OFF` 即可，tests 默认开启。）

## 可选：内嵌单文件模式

权重直接打进 exe，产出完全自包含的单文件（≈188 MB），运行时不需要 `-m`：

```powershell
& "$VSBT\CMake\bin\cmake.exe" -S . -B build-embed -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAIDASS_BUILD_TESTS=OFF -DHAIDASS_EMBED_MODEL="$PWD\models\haidass-translate-143m-q8_0.gguf"
& "$VSBT\CMake\bin\cmake.exe" --build build-embed -j
.\build-embed\haidass.exe --embed-info
```

MSVC 下通过 Windows `.rc` 资源内嵌（见 `cmake/embed_gguf.rc.in`）。

## 常见问题

- **`UnexpectedToken` / 语法报错**：在 PowerShell 里用了 cmd 语法（`set` / `%VAR%`）。两种方式二选一，别混用。
- **找不到 cmake**：确认用的是上面 BuildTools 自带的路径；也可以 `winget install Kitware.CMake` 装独立版。
- **执行策略错误**：`Set-ExecutionPolicy -Scope Process Bypass`（只对当前窗口生效）。
- **模型下载中断**：重跑 `python tools\download_model.py`，会从中断处续传。
