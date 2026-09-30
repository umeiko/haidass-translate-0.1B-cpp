// tensor.h — tensor view over raw (quantized) weight data, zero-copy.
#pragma once

#include <cstddef>
#include <cstdint>

namespace haidass {

// GGML tensor type ids (values match the GGUF/ggml spec, do not renumber).
enum DType : int32_t {
    DT_F32  = 0,
    DT_F16  = 1,
    DT_Q4_0 = 2,
    DT_Q8_0 = 8,
    DT_BF16 = 30,
};

// Quantization block geometry: every dtype is stored in blocks of
// `block_size` elements occupying `type_size` bytes.
inline int64_t dtype_block_size(int32_t t) {
    switch (t) {
        case DT_Q4_0: return 32;
        case DT_Q8_0: return 32;
        default:      return 1;
    }
}

inline int64_t dtype_type_size(int32_t t) {
    switch (t) {
        case DT_F32:  return 4;
        case DT_F16:  return 2;
        case DT_BF16: return 2;
        case DT_Q4_0: return 18;  // f16 scale + 16 bytes (32 x 4-bit)
        case DT_Q8_0: return 34;  // f16 scale + 32 bytes
        default:      return 0;
    }
}

inline const char* dtype_name(int32_t t) {
    switch (t) {
        case DT_F32:  return "f32";
        case DT_F16:  return "f16";
        case DT_BF16: return "bf16";
        case DT_Q4_0: return "q4_0";
        case DT_Q8_0: return "q8_0";
        default:      return "unknown";
    }
}

// Non-owning view into the mapped GGUF data section.
// Shape is stored in C order: ne[0] is the innermost (contiguous) dimension.
struct TensorView {
    const uint8_t* data = nullptr;
    int32_t  dtype  = DT_F32;
    int64_t  ne[4]  = {1, 1, 1, 1};
    int      n_dims = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int i = 0; i < n_dims; ++i) n *= ne[i];
        return n;
    }

    // Bytes occupied by one row (the contiguous dim ne[0]).
    int64_t row_bytes() const {
        const int64_t bs = dtype_block_size(dtype);
        return (ne[0] / bs) * dtype_type_size(dtype);
    }

    // Pointer to row `r` of a 2D weight matrix (ne[1] rows of ne[0] elements).
    const uint8_t* row(int64_t r) const { return data + r * row_bytes(); }

    int64_t total_bytes() const {
        int64_t rows = 1;
        for (int i = 1; i < n_dims; ++i) rows *= ne[i];
        return rows * row_bytes();
    }
};

} // namespace haidass
