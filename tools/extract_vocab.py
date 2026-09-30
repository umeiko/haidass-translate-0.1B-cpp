#!/usr/bin/env python3
"""extract_vocab.py — extract tokenizer vocab (tokens/scores/types) from the
official GGUF into a compact binary blob (tests/data/vocab.bin) that the
fixture generator and tests consume without downloading the full model.

Blob format (little-endian):
  magic "HVT1" | u32 n | i32 bos | i32 eos
  n x { u32 text_len | u8 token_type | f32 score | text bytes }

Usage:
  python tools/extract_vocab.py [--gguf models/haidass-translate-143m-q8_0.gguf]
                                [--out tests/data/vocab.bin]
"""
import argparse
import os
import struct
import urllib.request

from ggufutil import GgufReader

REPO = "umeiko/Haidass-Translate-143M-GGUF"
FILE = "haidass-translate-143m-q8_0.gguf"
HEAD_BYTES = 4 * 1024 * 1024  # metadata section is ~1.5 MB


def fetch_head(url: str, nbytes: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="models/haidass-translate-143m-q8_0.gguf")
    ap.add_argument("--out", default="tests/data/vocab.bin")
    ap.add_argument("--hf", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.gguf):
        with open(args.gguf, "rb") as f:
            head = f.read(HEAD_BYTES)
    else:
        host = "huggingface.co" if args.hf else "hf-mirror.com"
        head = fetch_head(f"https://{host}/{REPO}/resolve/main/{FILE}", HEAD_BYTES)

    g = GgufReader(head)
    tokens = g.meta["tokenizer.ggml.tokens"]
    scores = g.meta["tokenizer.ggml.scores"]
    types = g.meta["tokenizer.ggml.token_type"]
    bos = g.meta["tokenizer.ggml.bos_token_id"]
    eos = g.meta["tokenizer.ggml.eos_token_id"]
    assert len(tokens) == len(scores) == len(types)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"HVT1")
        f.write(struct.pack("<Iii", len(tokens), bos, eos))
        for text, score, ttype in zip(tokens, scores, types):
            b = text.encode("utf-8", "surrogateescape")
            f.write(struct.pack("<IBf", len(b), int(ttype), float(score)))
            f.write(b)
    print(f"wrote {args.out}: {len(tokens)} tokens, bos={bos} eos={eos}")


if __name__ == "__main__":
    main()
