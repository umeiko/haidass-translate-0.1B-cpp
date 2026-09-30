// test_util.h — tiny assertion helpers (no framework).
#pragma once

#include <cmath>
#include <cstdio>

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: CHECK(%s)\n", __FILE__,        \
                         __LINE__, #cond);                                   \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                \
    do {                                                                     \
        double _d = std::fabs((double)(a) - (double)(b));                    \
        if (_d > (tol)) {                                                    \
            std::fprintf(stderr,                                             \
                         "FAIL %s:%d: |%s - %s| = %g > %g (%g vs %g)\n",     \
                         __FILE__, __LINE__, #a, #b, _d, (double)(tol),      \
                         (double)(a), (double)(b));                          \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static int test_summary(const char* name) {
    if (g_failures == 0) {
        std::fprintf(stderr, "PASS %s\n", name);
        return 0;
    }
    std::fprintf(stderr, "FAILED %s: %d check(s) failed\n", name, g_failures);
    return 1;
}
