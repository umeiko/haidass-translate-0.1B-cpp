// gguf.h — minimal read-only GGUF (v3) parser. Little-endian hosts only.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

namespace haidass {

// GGUF metadata value types (subset of the spec, values match the spec).
enum GgufValueType : int32_t {
    GV_UINT8 = 0, GV_INT8 = 1, GV_UINT16 = 2, GV_INT16 = 3,
    GV_UINT32 = 4, GV_INT32 = 5, GV_FLOAT32 = 6, GV_BOOL = 7,
    GV_STRING = 8, GV_ARRAY = 9, GV_UINT64 = 10, GV_INT64 = 11, GV_FLOAT64 = 12,
};

struct GgufValue {
    int32_t type = -1;                 // scalar type, or GV_ARRAY
    int32_t elem_type = -1;            // element type when type == GV_ARRAY
    int64_t  i = 0;                    // integer/bool scalars
    double   f = 0.0;                  // float scalars
    std::string s;                     // string scalar
    std::vector<std::string> strs;     // array<string>
    std::vector<double> nums;          // array<numeric> (f32/i32/u32 all exact in double)
};

class GgufFile {
public:
    // Parse a GGUF image held in memory (mmap'd file or embedded blob).
    // The caller must keep `data` alive for the lifetime of this object.
    bool open(const uint8_t* data, size_t size, std::string* err);

    // --- metadata access ---
    const GgufValue* find(const std::string& key) const;
    uint32_t get_u32(const std::string& key, uint32_t fallback = 0) const;
    float    get_f32(const std::string& key, float fallback = 0.0f) const;
    bool     get_bool(const std::string& key, bool fallback = false) const;
    std::string get_str(const std::string& key, const std::string& fallback = "") const;

    // --- tensors ---
    const TensorView* tensor(const std::string& name) const;
    size_t num_tensors() const { return tensors_.size(); }

    int32_t version() const { return version_; }
    const std::string& arch() const { return arch_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    int32_t version_ = 0;
    std::string arch_;
    std::unordered_map<std::string, GgufValue> kv_;
    std::unordered_map<std::string, TensorView> tensors_;
};

} // namespace haidass
