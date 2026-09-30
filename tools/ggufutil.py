"""ggufutil.py — minimal pure-python GGUF v3 reader (metadata + tensor infos)
and writer (f32 tensors). No third-party deps for reading/writing structure.
"""
import struct

GGUF_VALUE_TYPES = {
    "u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5,
    "f32": 6, "bool": 7, "str": 8, "arr": 9, "u64": 10, "i64": 11, "f64": 12,
}

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 30


class GgufReader:
    """Parses header/metadata/tensor directory from a bytes buffer."""

    def __init__(self, data: bytes):
        self.data = data
        assert data[:4] == b"GGUF", "bad magic"
        self.version, self.n_tensors, self.n_kv = struct.unpack_from("<IQQ", data, 4)
        assert self.version == 3, f"unsupported version {self.version}"
        self.off = 24
        self.meta = {}
        for _ in range(self.n_kv):
            key = self._str()
            vtype = self._u32()
            self.meta[key] = self._value(vtype)
        self.tensors = []  # (name, dims[list, innermost first], ggml_type, offset)
        for _ in range(self.n_tensors):
            name = self._str()
            nd = self._u32()
            dims = list(struct.unpack_from("<%dQ" % nd, data, self.off))
            self.off += 8 * nd
            ttype = self._u32()
            toff = self._u64()
            self.tensors.append((name, dims, ttype, toff))
        alignment = self.meta.get("general.alignment", 32)
        self.data_start = (self.off + alignment - 1) // alignment * alignment

    def _u32(self):
        v = struct.unpack_from("<I", self.data, self.off)[0]
        self.off += 4
        return v

    def _u64(self):
        v = struct.unpack_from("<Q", self.data, self.off)[0]
        self.off += 8
        return v

    def _str(self):
        n = self._u64()
        s = self.data[self.off:self.off + n]
        self.off += n
        return s.decode("utf-8", "surrogateescape")

    def _value(self, vtype):
        scalar_fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                      6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if vtype == 8:
            return self._str()
        if vtype == 9:
            etype = self._u32()
            n = self._u64()
            return [self._value(etype) for _ in range(n)]
        fmt = scalar_fmt[vtype]
        v = struct.unpack_from(fmt, self.data, self.off)[0]
        self.off += struct.calcsize(fmt)
        return v


def _pack_value(v):
    """Returns (type_id, payload_bytes) for a python value."""
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, int):
        return 4, struct.pack("<I", v)
    if isinstance(v, str):
        b = v.encode("utf-8", "surrogateescape")
        return 8, struct.pack("<Q", len(b)) + b
    if isinstance(v, (list, tuple)):
        if v and isinstance(v[0], str):
            payload = struct.pack("<IQ", 8, len(v))
            for s in v:
                b = s.encode("utf-8", "surrogateescape")
                payload += struct.pack("<Q", len(b)) + b
            return 9, payload
        if v and isinstance(v[0], float):
            payload = struct.pack("<IQ", 6, len(v))
            payload += struct.pack("<%df" % len(v), *v)
            return 9, payload
        # integer array -> i32
        payload = struct.pack("<IQ", 5, len(v))
        payload += struct.pack("<%di" % len(v), *v)
        return 9, payload
    raise ValueError(f"unsupported metadata value type: {type(v)}")


class GgufWriter:
    """Writes a GGUF v3 file with f32 tensors.

    dims are given innermost-first (GGUF convention), i.e. a (rows, k) C-style
    matrix is dims=[k, rows].
    """

    def __init__(self, alignment=32):
        self.alignment = alignment
        self.meta = []    # list of (key, python value)
        self.tensors = []  # list of (name, dims, ggml_type, bytes)

    def add_meta(self, key, value):
        self.meta.append((key, value))

    def add_tensor(self, name, dims, raw: bytes, ggml_type=GGML_TYPE_F32):
        self.tensors.append((name, list(dims), ggml_type, bytes(raw)))

    def build(self) -> bytes:
        out = bytearray()
        out += b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.meta))
        for key, value in self.meta:
            kb = key.encode("utf-8")
            out += struct.pack("<Q", len(kb)) + kb
            vtype, payload = _pack_value(value)
            out += struct.pack("<I", vtype) + payload
        offset = 0
        infos = []
        for name, dims, gtype, raw in self.tensors:
            infos.append((name, dims, gtype, offset))
            offset += (offset + self.alignment - 1) // self.alignment * self.alignment - offset
            offset += len(raw)
        for (name, dims, gtype, off) in infos:
            nb = name.encode("utf-8")
            out += struct.pack("<Q", len(nb)) + nb
            out += struct.pack("<I", len(dims))
            out += struct.pack("<%dQ" % len(dims), *dims)
            out += struct.pack("<IQ", gtype, off)
        # data section
        pad = (len(out) + self.alignment - 1) // self.alignment * self.alignment - len(out)
        out += b"\x00" * pad
        cursor = 0
        for (name, dims, gtype, raw) in self.tensors:
            pad = (cursor + self.alignment - 1) // self.alignment * self.alignment - cursor
            if pad:
                out += b"\x00" * pad
                cursor += pad
            out += raw
            cursor += len(raw)
        return bytes(out)
