#!/usr/bin/env bash
# build_mingw.sh — no-CMake fallback build with a plain g++ (MinGW/MSYS2).
# Usage:  bash scripts/build_mingw.sh [build_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-out-mingw}"
mkdir -p "$OUT"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O3 -DNDEBUG -Isrc -Wall -fno-strict-aliasing"

CORE_SRCS="src/platform.cpp src/gguf.cpp src/tokenizer.cpp src/ops.cpp \
src/cpu_dispatch.cpp src/thread_pool.cpp src/kernels_scalar.cpp \
src/model.cpp src/sampler.cpp src/generate.cpp src/embed_none.cpp"

echo ">> core (portable baseline)"
for f in $CORE_SRCS; do
    obj="$OUT/$(basename "${f%.cpp}").o"
    echo "  CXX $f"
    $CXX $CXXFLAGS -DHAIDASS_HAVE_AVX2=1 -c "$f" -o "$obj"
done

echo ">> kernels_avx2 (own ISA flags)"
$CXX $CXXFLAGS -DHAIDASS_HAVE_AVX2=1 -mavx2 -mfma -mf16c -c src/kernels_avx2.cpp -o "$OUT/kernels_avx2.o"

echo ">> cli"
$CXX $CXXFLAGS -DHAIDASS_HAVE_AVX2=1 -c src/main.cpp -o "$OUT/main.o"

echo ">> link"
$CXX $CXXFLAGS "$OUT"/*.o -o "$OUT/haidass.exe" -static -lpthread
echo "done: $OUT/haidass.exe"
