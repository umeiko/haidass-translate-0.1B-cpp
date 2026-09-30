#!/usr/bin/env python3
"""make_fixture.py — build a tiny random-weight Qwen3 fixture GGUF for tests.

Uses the real 64k tokenizer vocab (tests/data/vocab.bin, produced by
tools/extract_vocab.py) but a 2-layer / hidden-64 model with deterministic
random f32 weights, so tests are fast and reproducible.

Usage:
  python tools/make_fixture.py --vocab tests/data/vocab.bin --out fixture.gguf
"""
import argparse
import struct

import numpy as np

from ggufutil import GgufWriter, GGML_TYPE_F32

import os

# fixture geometry (small but structurally identical to the real model)
LAYERS = 2
HIDDEN = 64
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 16
FFN = 128
CTX = 512
ROPE_THETA = 100000.0
RMS_EPS = 1e-6


def load_vocab(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"HVT1", "bad vocab blob"
    n, bos, eos = struct.unpack_from("<Iii", data, 4)
    off = 16
    tokens, scores, types = [], [], []
    for _ in range(n):
        (tlen,) = struct.unpack_from("<I", data, off)
        ttype, score = struct.unpack_from("<Bf", data, off + 4)
        off += 9
        text = data[off:off + tlen].decode("utf-8", "surrogateescape")
        off += tlen
        tokens.append(text)
        scores.append(score)
        types.append(ttype)
    return tokens, scores, types, bos, eos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="tests/data/vocab.bin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokens, scores, types, bos, eos = load_vocab(args.vocab)
    vocab = len(tokens)

    rng = np.random.RandomState(1234)

    def rand(rows, k):
        return (rng.randn(rows, k) * 0.02).astype(np.float32)

    def ones(n):
        return np.ones(n, dtype=np.float32)

    w = GgufWriter()
    w.add_meta("general.architecture", "qwen3")
    w.add_meta("general.name", "haidass-fixture")
    w.add_meta("qwen3.block_count", LAYERS)
    w.add_meta("qwen3.context_length", CTX)
    w.add_meta("qwen3.embedding_length", HIDDEN)
    w.add_meta("qwen3.feed_forward_length", FFN)
    w.add_meta("qwen3.attention.head_count", N_HEADS)
    w.add_meta("qwen3.attention.head_count_kv", N_KV_HEADS)
    w.add_meta("qwen3.attention.key_length", HEAD_DIM)
    w.add_meta("qwen3.attention.value_length", HEAD_DIM)
    w.add_meta("qwen3.rope.freq_base", ROPE_THETA)
    w.add_meta("qwen3.attention.layer_norm_rms_epsilon", RMS_EPS)
    w.add_meta("tokenizer.ggml.model", "llama")
    w.add_meta("tokenizer.ggml.pre", "default")
    w.add_meta("tokenizer.ggml.tokens", tokens)
    w.add_meta("tokenizer.ggml.scores", [float(s) for s in scores])
    w.add_meta("tokenizer.ggml.token_type", [int(t) for t in types])
    w.add_meta("tokenizer.ggml.bos_token_id", bos)
    w.add_meta("tokenizer.ggml.eos_token_id", eos)
    w.add_meta("tokenizer.ggml.add_bos_token", False)
    w.add_meta("tokenizer.ggml.add_eos_token", False)

    def add(name, arr):
        # arr is (rows, k) C-order; GGUF stores innermost first
        rows, k = arr.shape
        w.add_tensor(name, [k, rows], arr.astype("<f4").tobytes(), GGML_TYPE_F32)

    def add1d(name, arr):
        w.add_tensor(name, [arr.shape[0]], arr.astype("<f4").tobytes(), GGML_TYPE_F32)

    add("token_embd.weight", rand(vocab, HIDDEN))
    for l in range(LAYERS):
        p = f"blk.{l}."
        add1d(p + "attn_norm.weight", ones(HIDDEN))
        add(p + "attn_q.weight", rand(N_HEADS * HEAD_DIM, HIDDEN))
        add(p + "attn_k.weight", rand(N_KV_HEADS * HEAD_DIM, HIDDEN))
        add(p + "attn_v.weight", rand(N_KV_HEADS * HEAD_DIM, HIDDEN))
        add(p + "attn_output.weight", rand(HIDDEN, N_HEADS * HEAD_DIM))
        add1d(p + "attn_q_norm.weight", ones(HEAD_DIM))
        add1d(p + "attn_k_norm.weight", ones(HEAD_DIM))
        add1d(p + "ffn_norm.weight", ones(HIDDEN))
        add(p + "ffn_gate.weight", rand(FFN, HIDDEN))
        add(p + "ffn_up.weight", rand(FFN, HIDDEN))
        add(p + "ffn_down.weight", rand(HIDDEN, FFN))
    add1d("output_norm.weight", ones(HIDDEN))

    blob = w.build()
    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"wrote {args.out}: {len(blob)} bytes, vocab={vocab}")


if __name__ == "__main__":
    main()
