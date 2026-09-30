// cpu_dispatch.cpp — runtime CPU feature detection, installs best kernels.
#include "kernels.h"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#    define HAIDASS_X86 1
#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <cpuid.h>
#    endif
#endif

#if defined(__arm__) && !defined(__aarch64__) && defined(__linux__)
#    include <sys/auxv.h>
#    ifndef HWCAP_NEON
#        define HWCAP_NEON 4096
#    endif
#endif

namespace haidass {

Kernels g_kernels;
static const char* g_tier = "scalar";

#if defined(HAIDASS_X86)
static void cpuid_ex(int leaf, int sub, uint32_t out[4]) {
#    if defined(_MSC_VER)
    int r[4];
    __cpuidex(r, leaf, sub);
    for (int i = 0; i < 4; ++i) out[i] = (uint32_t)r[i];
#    else
    uint32_t a, b, c, d;
    __cpuid_count(leaf, sub, a, b, c, d);
    out[0] = a; out[1] = b; out[2] = c; out[3] = d;
#    endif
}

static uint64_t xgetbv0() {
#    if defined(_MSC_VER)
    return _xgetbv(0);
#    else
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#    endif
}

static bool cpu_has_avx2_fma_f16c() {
    uint32_t r[4];
    cpuid_ex(0, 0, r);
    const uint32_t max_leaf = r[0];
    if (max_leaf < 7) return false;
    cpuid_ex(1, 0, r);
    const bool osxsave = (r[2] & (1u << 27)) != 0;
    const bool avx = (r[2] & (1u << 28)) != 0;
    const bool fma = (r[2] & (1u << 12)) != 0;
    const bool f16c = (r[2] & (1u << 29)) != 0;
    if (!osxsave || !avx) return false;
    if ((xgetbv0() & 0x6) != 0x6) return false;  // OS saves YMM state
    cpuid_ex(7, 0, r);
    const bool avx2 = (r[1] & (1u << 5)) != 0;
    return avx2 && fma && f16c;
}
#endif  // HAIDASS_X86

void kernels_init() {
    kernels_scalar_fill(&g_kernels);
    g_tier = "scalar";

#if defined(HAIDASS_HAVE_AVX2) && defined(HAIDASS_X86)
    if (cpu_has_avx2_fma_f16c()) {
        kernels_avx2_fill(&g_kernels);
        g_tier = "avx2";
        return;
    }
#endif

#if defined(HAIDASS_HAVE_NEON)
#    if defined(__aarch64__)
    kernels_neon_fill(&g_kernels);
    g_tier = "neon";
    return;
#    elif defined(__arm__) && defined(__linux__)
    if (getauxval(AT_HWCAP) & HWCAP_NEON) {
        kernels_neon_fill(&g_kernels);
        g_tier = "neon";
        return;
    }
#    endif
#endif
}

const char* kernels_active_tier() { return g_tier; }

} // namespace haidass
