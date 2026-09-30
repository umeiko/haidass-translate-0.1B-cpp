// embed_res.cpp — embedded build via Windows resource (MSVC).
#include "platform.h"

#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>

#define IDR_GGUF 101

namespace haidass {

static HRSRC find_resource() {
    // RT_RCDATA follows the UNICODE define and may decay to LPSTR; spell out
    // the W form so this compiles regardless of the project character set.
    return FindResourceW(nullptr, MAKEINTRESOURCEW(IDR_GGUF), MAKEINTRESOURCEW(10 /*RT_RCDATA*/));
}

const uint8_t* embedded_model_data() {
    HRSRC rsrc = find_resource();
    if (!rsrc) return nullptr;
    HGLOBAL h = LoadResource(nullptr, rsrc);
    if (!h) return nullptr;
    return static_cast<const uint8_t*>(LockResource(h));
}

size_t embedded_model_size() {
    HRSRC rsrc = find_resource();
    if (!rsrc) return 0;
    return static_cast<size_t>(SizeofResource(nullptr, rsrc));
}

} // namespace haidass
