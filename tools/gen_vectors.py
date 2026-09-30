#!/usr/bin/env python3
"""gen_vectors.py — generate test vectors.

1) tokenizer_vectors.txt: expected token ids for a battery of strings,
   computed with an independent Python port of the same llama.cpp
   SPM-merge algorithm the C++ tokenizer implements.
2) logits_ref.bin: reference logits for a fixed prompt over the fixture
   model, computed with a numpy reference forward pass.
   Layout: u32 n_ids | u32 vocab | i32 ids[n_ids] | f32 logits[vocab].

Usage:
  python tools/gen_vectors.py --vocab tests/data/vocab.bin \
      --fixture build/tests/fixture.gguf \
      --tokenizer-vectors build/tests/tokenizer_vectors.txt \
      --logits-out build/tests/logits_ref.bin
"""
import argparse
import base64
import heapq
import struct

import numpy as np

from ggufutil import GgufReader

from make_fixture import load_vocab

SP = "▁"  # ▁

TEST_TEXTS = [
    "Machine translation bridges languages and cultures.",
    "The trial took place at Birmingham Crown Court and concluded on August 3.",
    "Hello, world! 123",
    "机器翻译连接了不同的语言与文化。",
    "审判于8月3日在伯明翰皇家法院进行,并于当日结束。",
    "人工智能正在改变世界,2026年是关键一年。",
    "Mixed 中英 bilingual text 混合输入 with numbers 42 and emoji 😀.",
    "  leading and trailing spaces  ",
    "Tabs\tand\nnewlines\nhere",
    "Specials in text <|im_start|> and <|im_end|> should split",
    "café naïve résumé über",
    "«Bonjour» — こんにちは — 안녕하세요",
    "",
    " ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "Quantiles: 3.14159, -2.71828, 1e-10",
]

PROMPT_TEXT = "Machine translation bridges languages and cultures."
PROMPT = ("<|im_start|>user\nTranslate the following text from English to Simplified Chinese.\n"
          + PROMPT_TEXT + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


class ReferenceTokenizer:
    """Python port of the llama.cpp llm_tokenizer_spm_session algorithm."""

    def __init__(self, tokens, scores, types, bos, eos):
        self.tokens = tokens
        self.scores = scores
        self.types = types
        self.bos = bos
        self.eos = eos
        # merge/lookup operates on raw UTF-8 bytes
        self.tok2id = {t.encode("utf-8", "surrogateescape"): i
                       for i, t in enumerate(tokens)}
        self.byte2id = {}
        for i, t in enumerate(tokens):
            if types[i] == 6 and len(t) == 6 and t.startswith("<0x"):
                try:
                    self.byte2id[int(t[3:5], 16)] = i
                except ValueError:
                    pass
        self.specials = sorted(
            (i for i in range(len(tokens)) if types[i] in (2, 3, 4)),
            key=lambda i: -len(tokens[i]))

    @staticmethod
    def _escape(text):
        return text.replace(" ", SP)

    def _tokenize_fragment(self, text, out):
        if not text:
            return
        # split into utf8 chars (byte lengths, like unicode_len_utf8)
        syms = []
        offs = 0
        while offs < len(text):
            c = text[offs]
            hb = c >> 4
            ln = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4)[hb]
            ln = min(ln, len(text) - offs)
            syms.append([offs, ln])
            offs += ln
        # linked list via prev/next
        n = len(syms)
        prev = [i - 1 for i in range(n)]
        nxt = [(i + 1) if i + 1 < n else -1 for i in range(n)]
        alive = [True] * n

        rev_merge = {}
        heap = []

        def text_of(i, j):
            a, la = syms[i]
            b, lb = syms[j]
            assert a + la == b
            return text[a:b + lb]

        def try_add(left, right):
            if left == -1 or right == -1:
                return
            t = text_of(left, right)
            tid = self.tok2id.get(t)
            if tid is None:
                return
            heapq.heappush(heap, (-self.scores[tid], left, right, len(t)))
            rev_merge[t] = (left, right)

        for i in range(1, n):
            try_add(i - 1, i)

        while heap:
            neg_score, left, right, size = heapq.heappop(heap)
            if not (alive[left] and alive[right]):
                continue
            if syms[left][1] + syms[right][1] != size:
                continue
            syms[left][1] += syms[right][1]
            alive[right] = False
            nx = nxt[right]
            nxt[left] = nx
            if nx >= 0:
                prev[nx] = left
            try_add(prev[left], left)
            try_add(left, nxt[left])

        def resegment(i):
            a, la = syms[i]
            t = text[a:a + la]
            tid = self.tok2id.get(t)
            if tid is not None:
                out.append(tid)
                return
            m = rev_merge.get(t)
            if m is None:
                for byte in t:
                    bid = self.byte2id.get(byte)
                    if bid is not None:
                        out.append(bid)
                return
            resegment(m[0])
            resegment(m[1])

        i = 0
        while i != -1:
            resegment(i)
            i = nxt[i]

    def encode(self, raw_text):
        out = []
        if not raw_text:
            return out
        # fragment split by specials (longest-first, plain substring search)
        frags = [(False, None, 0, len(raw_text))]
        for sid in self.specials:
            stext = self.tokens[sid]
            if not stext:
                continue
            new_frags = []
            for is_special, tok, off, ln in frags:
                if is_special:
                    new_frags.append((is_special, tok, off, ln))
                    continue
                base = off
                end = off + ln
                while True:
                    m = raw_text.find(stext, base)
                    if m == -1 or m + len(stext) > end:
                        break
                    if m > base:
                        new_frags.append((False, None, base, m - base))
                    new_frags.append((True, sid, m, len(stext)))
                    base = m + len(stext)
                if base < end:
                    new_frags.append((False, None, base, end - base))
            frags = new_frags
        is_prev_special = True
        for is_special, tok, off, ln in frags:
            if is_special:
                out.append(tok)
                is_prev_special = True
                continue
            text = raw_text[off:off + ln]
            if is_prev_special:
                text = " " + text
            text = self._escape(text)
            self._tokenize_fragment(text.encode("utf-8", "surrogateescape"), out)
            is_prev_special = False
        return out


def rmsnorm(x, w, eps):
    ss = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ss + eps) * w


def rope(v, half, cos_row, sin_row):
    v0 = v[..., :half]
    v1 = v[..., half:]
    return np.concatenate([v0 * cos_row - v1 * sin_row,
                           v1 * cos_row + v0 * sin_row], axis=-1)


def reference_forward(ids, g, weights):
    """numpy Qwen3 forward; returns logits of the last position (float32)."""
    meta = g.meta
    n_layers = meta["qwen3.block_count"]
    hidden = meta["qwen3.embedding_length"]
    n_heads = meta["qwen3.attention.head_count"]
    n_kv = meta["qwen3.attention.head_count_kv"]
    hd = meta["qwen3.attention.key_length"]
    eps = meta["qwen3.attention.layer_norm_rms_epsilon"]
    theta = meta["qwen3.rope.freq_base"]
    half = hd // 2

    T = len(ids)
    embd = weights["token_embd.weight"]
    x = embd[ids].astype(np.float64)  # (T, hidden)

    inv_freq = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hd)
    angles = np.outer(np.arange(T), inv_freq)
    cos_t, sin_t = np.cos(angles), np.sin(angles)

    kcache = np.zeros((n_layers, T, n_kv, hd))
    vcache = np.zeros((n_layers, T, n_kv, hd))

    for l in range(n_layers):
        p = f"blk.{l}."
        xn = rmsnorm(x, weights[p + "attn_norm.weight"], eps)
        q = (xn @ weights[p + "attn_q.weight"].T).reshape(T, n_heads, hd)
        k = (xn @ weights[p + "attn_k.weight"].T).reshape(T, n_kv, hd)
        v = (xn @ weights[p + "attn_v.weight"].T).reshape(T, n_kv, hd)
        qn = weights[p + "attn_q_norm.weight"]
        kn = weights[p + "attn_k_norm.weight"]
        for t in range(T):
            for h in range(n_heads):
                q[t, h] = rope(rmsnorm(q[t, h], qn, eps), half, cos_t[t], sin_t[t])
            for h in range(n_kv):
                k[t, h] = rope(rmsnorm(k[t, h], kn, eps), half, cos_t[t], sin_t[t])
        kcache[l] = k
        vcache[l] = v
        group = n_heads // n_kv
        out = np.zeros((T, n_heads, hd))
        for t in range(T):
            for h in range(n_heads):
                kvh = h // group
                scores = (kcache[l, :t + 1, kvh, :] @ q[t, h]) / np.sqrt(hd)
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                out[t, h] = scores @ vcache[l, :t + 1, kvh, :]
        x = x + out.reshape(T, n_heads * hd) @ weights[p + "attn_output.weight"].T

        xn = rmsnorm(x, weights[p + "ffn_norm.weight"], eps)
        gate = xn @ weights[p + "ffn_gate.weight"].T
        up = xn @ weights[p + "ffn_up.weight"].T
        h = gate / (1.0 + np.exp(-gate)) * up
        x = x + h @ weights[p + "ffn_down.weight"].T

    xn = rmsnorm(x, weights["output_norm.weight"], eps)
    logits = xn[-1] @ embd.T
    return logits.astype(np.float32)


def load_fixture_weights(path):
    with open(path, "rb") as f:
        data = f.read()
    g = GgufReader(data)
    weights = {}
    for name, dims, gtype, off in g.tensors:
        assert gtype == 0, f"fixture tensor {name} not f32"
        count = 1
        for d in dims:
            count *= d
        arr = np.frombuffer(data, dtype="<f4", count=count,
                            offset=g.data_start + off)
        # GGUF dims are innermost-first; C-order array shape is reversed
        weights[name] = arr.reshape(list(reversed(dims)))
    return g, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="tests/data/vocab.bin")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tokenizer-vectors", required=True)
    ap.add_argument("--logits-out", required=True)
    args = ap.parse_args()

    tokens, scores, types, bos, eos = load_vocab(args.vocab)
    rt = ReferenceTokenizer(tokens, scores, types, bos, eos)

    with open(args.tokenizer_vectors, "w", encoding="utf-8") as f:
        for t in TEST_TEXTS:
            ids = rt.encode(t)
            b64 = base64.b64encode(t.encode("utf-8")).decode("ascii")
            f.write(b64 + "\t" + ",".join(str(i) for i in ids) + "\n")
    print(f"wrote {args.tokenizer_vectors}: {len(TEST_TEXTS)} vectors")

    g, weights = load_fixture_weights(args.fixture)
    ids = rt.encode(PROMPT)
    assert len(ids) > 0
    logits = reference_forward(ids, g, weights)
    with open(args.logits_out, "wb") as f:
        f.write(struct.pack("<II", len(ids), logits.shape[0]))
        f.write(struct.pack("<%di" % len(ids), *ids))
        f.write(logits.astype("<f4").tobytes())
    print(f"wrote {args.logits_out}: {len(ids)} prompt ids, vocab={logits.shape[0]}")
    top = int(np.argmax(logits))
    print(f"ref top-1 next token id: {top} ({tokens[top]!r})")


if __name__ == "__main__":
    main()
