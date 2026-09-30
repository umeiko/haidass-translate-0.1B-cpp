#!/usr/bin/env python3
"""download_model.py — download the official Q8_0 GGUF (default: hf-mirror).

Usage:
  python tools/download_model.py [--out models] [--hf]   # --hf: huggingface.co
"""
import argparse
import os
import sys
import urllib.request

REPO = "umeiko/Haidass-Translate-143M-GGUF"
FILE = "haidass-translate-143m-q8_0.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models")
    ap.add_argument("--hf", action="store_true", help="use huggingface.co instead of hf-mirror.com")
    args = ap.parse_args()

    host = "huggingface.co" if args.hf else "hf-mirror.com"
    url = f"https://{host}/{REPO}/resolve/main/{FILE}"
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, FILE)

    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab") as f:
        total = have
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r{total/1e6:.1f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    os.replace(tmp, dest)
    print(f"saved: {dest} ({os.path.getsize(dest)} bytes)")


if __name__ == "__main__":
    main()
