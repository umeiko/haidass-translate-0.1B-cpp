// embed_asm.cpp — embedded build via assembler .incbin (ELF / Mach-O / MinGW).
#include "platform.h"

extern "C" {
#if defined(__APPLE__)
extern const uint8_t _gguf_embed_start[];
extern const uint8_t _gguf_embed_end[];
#else
extern const uint8_t gguf_embed_start[];
extern const uint8_t gguf_embed_end[];
#endif
}

namespace haidass {

const uint8_t* embedded_model_data() {
#if defined(__APPLE__)
    return _gguf_embed_start;
#else
    return gguf_embed_start;
#endif
}

size_t embedded_model_size() {
#if defined(__APPLE__)
    return static_cast<size_t>(_gguf_embed_end - _gguf_embed_start);
#else
    return static_cast<size_t>(gguf_embed_end - gguf_embed_start);
#endif
}

} // namespace haidass
