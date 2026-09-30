#!/usr/bin/env python3
"""verify_vs_hf.py — ground-truth comparison against HF transformers.

Loads the SAME local Q8_0 GGUF via transformers' GGUF integration and
greedy-decodes the test battery; also dumps HF reference logits for one
prompt (same .bin layout as gen_vectors.py) for numeric comparison with
tests/forward_logits_test.

Usage:
  python tools/verify_vs_hf.py --gguf models/haidass-translate-143m-q8_0.gguf \
      --hf-dir models/hf --logits-bin build/hf_logits.bin
"""
import argparse
import os
import struct
import sys
import urllib.request

import torch

DALAB = "DALabCommunity/Haidass-Translate-143M"
HF_FILES = ["config.json", "tokenizer.model", "tokenizer_config.json",
            "special_tokens_map.json"]

CASES = [
    (True, "Machine translation bridges languages and cultures."),
    (True, "The trial took place at Birmingham Crown Court and concluded on August 3."),
    (True, "Artificial intelligence is reshaping the world."),
    (False, "机器翻译连接了不同的语言与文化。"),
    (False, "审判于8月3日在伯明翰皇家法院进行,并于当日结束。"),
    (False, "人工智能正在改变世界,2026年是关键一年。"),
]

PROMPT_EN = ("<|im_start|>user\nTranslate the following text from English to Simplified Chinese.\n"
             "{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
PROMPT_ZH = ("<|im_start|>user\n请将以下简体中文翻译成英文。\n"
             "{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def ensure_hf_files(hf_dir):
    os.makedirs(hf_dir, exist_ok=True)
    for name in HF_FILES:
        dest = os.path.join(hf_dir, name)
        if os.path.exists(dest):
            continue
        url = f"https://hf-mirror.com/{DALAB}/resolve/main/{name}"
        print(f"fetch {url}")
        urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="models/haidass-translate-143m-q8_0.gguf")
    ap.add_argument("--hf-dir", default="models/hf")
    ap.add_argument("--logits-bin", default=None)
    ap.add_argument("--vocab", default="tests/data/vocab.bin")
    args = ap.parse_args()

    ensure_hf_files(args.hf_dir)

    # Tokenize with the independent Python port of the llama.cpp SPM session
    # (gen_vectors.ReferenceTokenizer) — the same algorithm the C++ engine
    # implements. Native sentencepiece CANNOT encode this chat prompt
    # correctly: it folds '\n' into whitespace and adds the dummy ▁ prefix
    # only once at the very start, while this model expects explicit <0x0A>
    # byte tokens and a ▁ prefix on EVERY raw-text segment between special
    # tokens (llama.cpp semantics). Verified empirically: with the native SP
    # encoding the model emits <|im_end|> immediately; with ours it translates.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_vectors import ReferenceTokenizer
    from make_fixture import load_vocab
    tokens, scores, types, bos_id, eos_id = load_vocab(args.vocab)
    rt = ReferenceTokenizer(tokens, scores, types, bos_id, eos_id)

    import sentencepiece as spm  # decode-only
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(args.hf_dir, "tokenizer.model"))

    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

    # config.json is authoritative (head_dim=64); the transformers GGUF config
    # mapping gets head_dim wrong for this model, so build from config + load
    # GGUF tensors manually.
    cfg = AutoConfig.from_pretrained(args.hf_dir)
    model = Qwen3ForCausalLM(cfg)
    ckpt = load_gguf_checkpoint(os.path.abspath(args.gguf), return_tensors=True,
                                model_to_load=model)
    missing, unexpected = model.load_state_dict(ckpt["tensors"], strict=False)
    if unexpected:
        print(f"WARNING unexpected tensors: {unexpected}")
    missing = [m for m in missing if m != "lm_head.weight"]  # tied embedding
    if missing:
        print(f"WARNING missing tensors: {missing}")
    model.tie_weights()
    model.eval()

    def encode(text):
        return torch.tensor([rt.encode(text)], dtype=torch.long)

    def decode(ids):
        return sp.decode([int(i) for i in ids])

    for en2zh, text in CASES:
        prompt = (PROMPT_EN if en2zh else PROMPT_ZH).format(text=text)
        ids = encode(prompt)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=256, do_sample=False,
                                 eos_token_id=5, pad_token_id=5)
        reply = decode(out[0][ids.shape[1]:])
        direction = "en->zh" if en2zh else "zh->en"
        print(f"[{direction}] {text}")
        print(f"   HF: {reply}")

    if args.logits_bin:
        text = "Machine translation bridges languages and cultures."
        prompt = PROMPT_EN.format(text=text)
        ids = encode(prompt)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float().numpy()
        id_list = ids[0].tolist()
        with open(args.logits_bin, "wb") as f:
            f.write(struct.pack("<II", len(id_list), logits.shape[0]))
            f.write(struct.pack("<%di" % len(id_list), *id_list))
            f.write(logits.astype("<f4").tobytes())
        print(f"wrote {args.logits_bin}: {len(id_list)} ids, vocab={logits.shape[0]}")


if __name__ == "__main__":
    main()
