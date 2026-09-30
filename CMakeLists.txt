cmake_minimum_required(VERSION 3.20)
project(haidass-translate LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(HAIDASS_BUILD_CLI   "Build the haidass CLI"                 ON)
option(HAIDASS_BUILD_TESTS "Build tests"                           ON)
option(HAIDASS_SIMD        "Enable SIMD kernels (AVX2/NEON)"       ON)
set(HAIDASS_EMBED_MODEL "" CACHE FILEPATH
    "Path to a .gguf model to embed into the binary (single-file mode)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(haidass_core STATIC
    src/platform.cpp
    src/gguf.cpp
    src/tokenizer.cpp
    src/ops.cpp
    src/cpu_dispatch.cpp
    src/thread_pool.cpp
    src/kernels_scalar.cpp
    src/model.cpp
    src/sampler.cpp
    src/generate.cpp
)
target_include_directories(haidass_core PUBLIC src)

find_package(Threads REQUIRED)
target_link_libraries(haidass_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# SIMD kernels: separate translation units with their own ISA flags.
# Runtime dispatch picks the best available at load time.
# ---------------------------------------------------------------------------
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" HAIDASS_ARCH)

if(HAIDASS_SIMD)
    if(HAIDASS_ARCH MATCHES "x86_64|amd64|x64|i686|i386|x86")
        add_library(haidass_kernels_avx2 STATIC src/kernels_avx2.cpp)
        target_include_directories(haidass_kernels_avx2 PRIVATE src)
        target_compile_definitions(haidass_kernels_avx2 PRIVATE HAIDASS_HAVE_AVX2=1)
        if(MSVC)
            target_compile_options(haidass_kernels_avx2 PRIVATE /arch:AVX2)
        else()
            target_compile_options(haidass_kernels_avx2 PRIVATE -mavx2 -mfma -mf16c)
        endif()
        target_link_libraries(haidass_core PUBLIC haidass_kernels_avx2)
        target_compile_definitions(haidass_core PRIVATE HAIDASS_HAVE_AVX2=1)
    elseif(HAIDASS_ARCH MATCHES "aarch64|arm64|arm")
        add_library(haidass_kernels_neon STATIC src/kernels_neon.cpp)
        target_include_directories(haidass_kernels_neon PRIVATE src)
        target_compile_definitions(haidass_kernels_neon PRIVATE HAIDASS_HAVE_NEON=1)
        # arm64: NEON is mandatory, no extra flags; armhf needs explicit FPU
        if(HAIDASS_ARCH MATCHES "^arm" AND NOT HAIDASS_ARCH MATCHES "aarch64|arm64")
            target_compile_options(haidass_kernels_neon PRIVATE -mfpu=neon)
        endif()
        target_link_libraries(haidass_core PUBLIC haidass_kernels_neon)
        target_compile_definitions(haidass_core PRIVATE HAIDASS_HAVE_NEON=1)
    endif()
endif()

# ---------------------------------------------------------------------------
# Embedded model (single-file mode)
# ---------------------------------------------------------------------------
if(HAIDASS_EMBED_MODEL)
    get_filename_component(HAIDASS_EMBED_MODEL_ABS "${HAIDASS_EMBED_MODEL}" ABSOLUTE)
    if(NOT EXISTS "${HAIDASS_EMBED_MODEL_ABS}")
        message(FATAL_ERROR "HAIDASS_EMBED_MODEL not found: ${HAIDASS_EMBED_MODEL_ABS}")
    endif()

    if(MSVC)
        # Windows resource embedding (compiler-independent format)
        file(TO_CMAKE_PATH "${HAIDASS_EMBED_MODEL_ABS}" HAIDASS_EMBED_MODEL_RC)
        configure_file(cmake/embed_gguf.rc.in ${CMAKE_BINARY_DIR}/embed_gguf.rc @ONLY)
        target_sources(haidass_core PRIVATE ${CMAKE_BINARY_DIR}/embed_gguf.rc src/embed_res.cpp)
    else()
        # GNU as .incbin works for ELF (Linux, MinGW) and Mach-O (macOS/clang)
        enable_language(ASM)
        file(TO_CMAKE_PATH "${HAIDASS_EMBED_MODEL_ABS}" HAIDASS_EMBED_MODEL_ASM)
        configure_file(cmake/embed_gguf.S.in ${CMAKE_BINARY_DIR}/embed_gguf.S @ONLY)
        target_sources(haidass_core PRIVATE ${CMAKE_BINARY_DIR}/embed_gguf.S src/embed_asm.cpp)
    endif()
    target_compile_definitions(haidass_core PRIVATE HAIDASS_EMBEDDED=1)
    message(STATUS "Embedded model: ${HAIDASS_EMBED_MODEL_ABS}")
else()
    target_sources(haidass_core PRIVATE src/embed_none.cpp)
endif()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if(HAIDASS_BUILD_CLI)
    add_executable(haidass src/main.cpp)
    target_link_libraries(haidass PRIVATE haidass_core)
    if(MSVC)
        target_compile_options(haidass PRIVATE /utf-8)
        target_compile_options(haidass_core PRIVATE /utf-8)
    endif()
    install(TARGETS haidass RUNTIME DESTINATION bin)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(HAIDASS_BUILD_TESTS)
    enable_testing()
    foreach(t gguf_test tokenizer_test ops_test forward_logits_test e2e_fixture_test)
        add_executable(${t} tests/${t}.cpp)
        target_link_libraries(${t} PRIVATE haidass_core)
        if(MSVC)
            target_compile_options(${t} PRIVATE /utf-8)
        endif()
    endforeach()

    # fixture + vectors generation (python + numpy)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
        add_test(NAME gen_fixture
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/make_fixture.py
                    --vocab ${CMAKE_SOURCE_DIR}/tests/data/vocab.bin
                    --out ${CMAKE_BINARY_DIR}/tests/fixture.gguf)
        add_test(NAME gen_vectors
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/gen_vectors.py
                    --vocab ${CMAKE_SOURCE_DIR}/tests/data/vocab.bin
                    --fixture ${CMAKE_BINARY_DIR}/tests/fixture.gguf
                    --tokenizer-vectors ${CMAKE_BINARY_DIR}/tests/tokenizer_vectors.txt
                    --logits-out ${CMAKE_BINARY_DIR}/tests/logits_ref.bin)
        set_tests_properties(gen_vectors PROPERTIES DEPENDS gen_fixture)

        add_test(NAME gguf_test COMMAND gguf_test ${CMAKE_BINARY_DIR}/tests/fixture.gguf)
        set_tests_properties(gguf_test PROPERTIES DEPENDS gen_fixture)
        add_test(NAME tokenizer_test
            COMMAND tokenizer_test ${CMAKE_BINARY_DIR}/tests/fixture.gguf
                    ${CMAKE_BINARY_DIR}/tests/tokenizer_vectors.txt)
        set_tests_properties(tokenizer_test PROPERTIES DEPENDS gen_vectors)
        add_test(NAME forward_logits_test
            COMMAND forward_logits_test ${CMAKE_BINARY_DIR}/tests/fixture.gguf
                    ${CMAKE_BINARY_DIR}/tests/logits_ref.bin)
        set_tests_properties(forward_logits_test PROPERTIES DEPENDS gen_vectors)
        add_test(NAME e2e_fixture_test COMMAND e2e_fixture_test ${CMAKE_BINARY_DIR}/tests/fixture.gguf)
        set_tests_properties(e2e_fixture_test PROPERTIES DEPENDS gen_fixture)
        add_test(NAME ops_test COMMAND ops_test)
    else()
        message(WARNING "Python3 not found: tests disabled")
    endif()
endif()
