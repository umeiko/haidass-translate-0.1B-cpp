// gguf.cpp — minimal read-only GGUF (v3) parser.
//
// Layout: header { "GGUF", u32 version, u64 n_tensors, u64 n_kv },
// then n_kv metadata pairs, then n_tensors tensor infos
// { name, u32 n_dims, u64 dims[n_dims] (innermost first), u32 type, u64 offset },
// then the data section, aligned to `general.alignment` (default 32).
// Tensor offsets are relative to the data section start.

#include "gguf.h"

#include <cstring>

namespace haidass {

namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;

    bool bytes(void* dst, size_t n) {
        if (static_cast<size_t>(end - p) < n) return false;
        std::memcpy(dst, p, n);
        p += n;
        return true;
    }

    template <typename T>
    bool pod(T& out) {
        if (static_cast<size_t>(end - p) < sizeof(T)) return false;
        std::memcpy(&out, p, sizeof(T));  // GGUF is little-endian; targets are LE
        p += sizeof(T);
        return true;
    }

    bool str(std::string& out) {
        uint64_t n = 0;
        if (!pod(n)) return false;
        if (n > static_cast<uint64_t>(end - p)) return false;
        out.assign(reinterpret_cast<const char*>(p), static_cast<size_t>(n));
        p += n;
        return true;
    }
};

bool read_value(Reader& r, int32_t vtype, GgufValue& v, int depth) {
    if (depth > 4) return false;
    v.type = vtype;
    switch (vtype) {
        case GV_UINT8: { uint8_t x;  if (!r.pod(x)) return false; v.i = x; return true; }
        case GV_INT8: { int8_t x;    if (!r.pod(x)) return false; v.i = x; return true; }
        case GV_UINT16: { uint16_t x; if (!r.pod(x)) return false; v.i = x; return true; }
        case GV_INT16: { int16_t x;  if (!r.pod(x)) return false; v.i = x; return true; }
        case GV_UINT32: { uint32_t x; if (!r.pod(x)) return false; v.i = x; return true; }
        case GV_INT32: { int32_t x;  if (!r.pod(x)) return false; v.i = x; return true; }
        case GV_UINT64: { uint64_t x; if (!r.pod(x)) return false; v.i = static_cast<int64_t>(x); return true; }
        case GV_INT64: { int64_t x;  if (!r.pod(x)) return false; v.i = x; return true; }
        case GV_FLOAT32: { float x;  if (!r.pod(x)) return false; v.f = x; return true; }
        case GV_FLOAT64: { double x; if (!r.pod(x)) return false; v.f = x; return true; }
        case GV_BOOL: { uint8_t x;   if (!r.pod(x)) return false; v.i = (x != 0); return true; }
        case GV_STRING: return r.str(v.s);
        case GV_ARRAY: {
            uint32_t etype = 0;
            uint64_t n = 0;
            if (!r.pod(etype) || !r.pod(n)) return false;
            if (n > (1ull << 32)) return false;
            v.elem_type = static_cast<int32_t>(etype);
            if (etype == GV_STRING) {
                v.strs.reserve(static_cast<size_t>(n));
                for (uint64_t i = 0; i < n; ++i) {
                    std::string s;
                    if (!r.str(s)) return false;
                    v.strs.push_back(std::move(s));
                }
                return true;
            }
            v.nums.reserve(static_cast<size_t>(n));
            for (uint64_t i = 0; i < n; ++i) {
                GgufValue e;
                if (!read_value(r, static_cast<int32_t>(etype), e, depth + 1)) return false;
                if (etype == GV_FLOAT32 || etype == GV_FLOAT64) {
                    v.nums.push_back(e.f);
                } else {
                    v.nums.push_back(static_cast<double>(e.i));
                }
            }
            return true;
        }
        default: return false;
    }
}

} // namespace

bool GgufFile::open(const uint8_t* data, size_t size, std::string* err) {
    data_ = data;
    size_ = size;
    kv_.clear();
    tensors_.clear();

    if (!data || size < 24) {
        if (err) *err = "file too small";
        return false;
    }
    if (std::memcmp(data, "GGUF", 4) != 0) {
        if (err) *err = "bad magic (not a GGUF file)";
        return false;
    }

    Reader r{data + 4, data + size};
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!r.pod(version) || !r.pod(n_tensors) || !r.pod(n_kv)) {
        if (err) *err = "truncated header";
        return false;
    }
    if (version != 3) {
        if (err) *err = "unsupported GGUF version " + std::to_string(version);
        return false;
    }
    version_ = static_cast<int32_t>(version);

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t vtype = 0;
        if (!r.str(key) || !r.pod(vtype)) {
            if (err) *err = "truncated metadata";
            return false;
        }
        GgufValue v;
        if (!read_value(r, static_cast<int32_t>(vtype), v, 0)) {
            if (err) *err = "bad metadata value for key: " + key;
            return false;
        }
        kv_[std::move(key)] = std::move(v);
    }

    uint32_t alignment = get_u32("general.alignment", 32);
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) alignment = 32;

    struct Info { std::string name; int32_t type; uint64_t offset; int64_t ne[4]; int n_dims; };
    std::vector<Info> infos;
    infos.reserve(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        Info info{};
        uint32_t n_dims = 0;
        if (!r.str(info.name) || !r.pod(n_dims) || n_dims > 4) {
            if (err) *err = "truncated tensor info";
            return false;
        }
        info.n_dims = static_cast<int>(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t ne = 0;
            if (!r.pod(ne)) {
                if (err) *err = "truncated tensor dims";
                return false;
            }
            info.ne[d] = static_cast<int64_t>(ne);
        }
        uint32_t type = 0;
        if (!r.pod(type) || !r.pod(info.offset)) {
            if (err) *err = "truncated tensor info";
            return false;
        }
        info.type = static_cast<int32_t>(type);
        infos.push_back(std::move(info));
    }

    const uint64_t data_start = ((uint64_t)(r.p - data) + alignment - 1) / alignment * alignment;
    if (data_start > size) {
        if (err) *err = "data section out of range";
        return false;
    }

    for (const Info& info : infos) {
        TensorView tv;
        tv.dtype = info.type;
        tv.n_dims = info.n_dims;
        for (int d = 0; d < 4; ++d) tv.ne[d] = d < info.n_dims ? info.ne[d] : 1;
        const int64_t bs = dtype_block_size(info.type);
        if (dtype_type_size(info.type) == 0 || (bs > 1 && tv.ne[0] % bs != 0)) {
            if (err) *err = "unsupported dtype/bad shape for tensor: " + info.name;
            return false;
        }
        const uint64_t nbytes = static_cast<uint64_t>(tv.total_bytes());
        if (info.offset + nbytes > size - data_start) {
            if (err) *err = "tensor data out of range: " + info.name;
            return false;
        }
        tv.data = data + data_start + info.offset;
        tensors_[info.name] = tv;
    }

    arch_ = get_str("general.architecture", "");
    return true;
}

const GgufValue* GgufFile::find(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
}

uint32_t GgufFile::get_u32(const std::string& key, uint32_t fallback) const {
    const GgufValue* v = find(key);
    if (!v) return fallback;
    if (v->type == GV_FLOAT32 || v->type == GV_FLOAT64) return static_cast<uint32_t>(v->f);
    return static_cast<uint32_t>(v->i);
}

float GgufFile::get_f32(const std::string& key, float fallback) const {
    const GgufValue* v = find(key);
    if (!v) return fallback;
    if (v->type == GV_FLOAT32 || v->type == GV_FLOAT64) return static_cast<float>(v->f);
    return static_cast<float>(v->i);
}

bool GgufFile::get_bool(const std::string& key, bool fallback) const {
    const GgufValue* v = find(key);
    if (!v) return fallback;
    return v->i != 0;
}

std::string GgufFile::get_str(const std::string& key, const std::string& fallback) const {
    const GgufValue* v = find(key);
    if (!v || v->type != GV_STRING) return fallback;
    return v->s;
}

const TensorView* GgufFile::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

} // namespace haidass
