#!/usr/bin/env python3
"""Dump llama-quantize --tensor-type-file from a GGUF (local or HF), headers only.

Usage:
  python3 extract_gguf_tensor_map.py /path/to/model-00001-of-00004.gguf -o unsloth.tensor-types
  python3 extract_gguf_tensor_map.py unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS -o unsloth.tensor-types
"""
from __future__ import annotations

import argparse, json, struct, sys, urllib.request
from collections import Counter
from pathlib import Path

MAGIC = b"GGUF"
SKIP_TYPES = {"F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64"}
GGML = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "IQ1_M",
    25: "BF16", 26: "Q4_0_4_4", 27: "Q4_0_4_8", 28: "Q4_0_8_8",
    29: "TQ1_0", 30: "TQ2_0", 32: "MXFP4", 36: "IQ1_BN", 37: "IQ2_BN",
}

def _u32(b, o): return struct.unpack_from("<I", b, o)[0], o + 4
def _u64(b, o): return struct.unpack_from("<Q", b, o)[0], o + 8
def _i32(b, o): return struct.unpack_from("<i", b, o)[0], o + 4

def _need(buf: bytearray, n: int, fetch):
    while len(buf) < n:
        chunk = fetch(len(buf), max(1 << 20, n - len(buf)))
        if not chunk:
            raise EOFError(f"short GGUF header, have {len(buf)} need {n}")
        buf.extend(chunk)

def parse_tensor_infos(fetch) -> list[tuple[str, str]]:
    buf = bytearray()
    _need(buf, 24, fetch)
    if buf[:4] != MAGIC:
        raise ValueError("not a GGUF")
    n_tensors, _ = _u64(buf, 8)
    n_kv, _ = _u64(buf, 16)
    o = 24

    def more(need):
        _need(buf, need, fetch)

    def skip_val(vt):
        nonlocal o
        if vt in (0, 1, 7):
            o += 1
        elif vt in (2, 3):
            o += 2
        elif vt in (4, 5, 6):
            o += 4
        elif vt in (10, 11, 12):
            o += 8
        elif vt == 8:
            more(o + 8)
            ln, o = _u64(buf, o)
            more(o + ln)
            o += ln
        elif vt == 9:
            more(o + 12)
            inner, o = _i32(buf, o)
            cnt, o = _u64(buf, o)
            for _ in range(cnt):
                more(o + 64)
                skip_val(inner)
        else:
            raise ValueError(f"unknown GGUF value type {vt} @ {o}")

    for _ in range(n_kv):
        more(o + 8)
        ln, o = _u64(buf, o)
        more(o + ln + 4)
        o += ln
        vt, o = _i32(buf, o)
        more(o + 16)
        skip_val(vt)

    out = []
    for _ in range(n_tensors):
        more(o + 8)
        ln, o = _u64(buf, o)
        more(o + ln + 4 + 8 * 5 + 4 + 8)
        name = bytes(buf[o:o + ln]).decode("utf-8")
        o += ln
        nd, o = _u32(buf, o)
        for _d in range(nd):
            more(o + 8)
            o += 8
        tid, o = _i32(buf, o)
        o += 8
        tname = GGML.get(tid, f"TYPE_{tid}")
        out.append((name, tname))
    return out

def fetch_local(path: Path):
    f = open(path, "rb")
    def _f(off, n):
        f.seek(off)
        return f.read(n)
    return _f, f.close

def fetch_http(url: str):
    def _f(off, n):
        req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + n - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    return _f, lambda: None

def _hf_tree(repo: str, path: str = "") -> list:
    url = f"https://huggingface.co/api/models/{repo}/tree/main/{path}" if path else f"https://huggingface.co/api/models/{repo}/tree/main"
    req = urllib.request.Request(url, headers={"User-Agent": "gguf-map/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def hf_files(repo: str, quant: str) -> list[str]:
    q = quant.lower()
    names = []
    for path in (quant, ""):
        try:
            items = _hf_tree(repo, path)
        except Exception as e:
            print(f"# tree {repo}/{path}: {e}", file=sys.stderr)
            continue
        for i in items:
            p = i.get("path") or ""
            if not str(p).endswith(".gguf"):
                continue
            low = p.lower()
            if "mmproj" in low or "mtp" in low:
                continue
            if q in low or (path == quant):
                names.append(p)
        if names:
            return sorted(set(names))
    raise FileNotFoundError(f"no GGUF for {repo}:{quant}")

def collect(src: str) -> list[tuple[str, str]]:
    if ":" in src and not Path(src).exists() and "/" in src.split(":")[0]:
        repo, quant = src.split(":", 1)
        files = hf_files(repo, quant)
        pairs = []
        for rel in files:
            url = f"https://huggingface.co/{repo}/resolve/main/{rel}"
            print(f"# header {url}", file=sys.stderr)
            fetch, close = fetch_http(url)
            try:
                pairs.extend(parse_tensor_infos(fetch))
            finally:
                close()
        return pairs
    p = Path(src)
    if p.is_dir():
        files = sorted(p.rglob("*.gguf"))
        files = [f for f in files if "mmproj" not in f.name.lower() and "mtp" not in f.name.lower()]
    elif p.is_file():
        files = [p]
        if "-of-" in p.name:
            files = sorted(p.parent.glob(p.name.split("-000")[0] + "-*.gguf")) or [p]
    else:
        raise FileNotFoundError(src)
    pairs = []
    for f in files:
        print(f"# header {f}", file=sys.stderr)
        fetch, close = fetch_local(f)
        try:
            pairs.extend(parse_tensor_infos(fetch))
        finally:
            close()
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="local GGUF/dir or repo:QUANT (unsloth/Qwen3.8-Flash-Next-GGUF:UD-IQ4_XS)")
    ap.add_argument("-o", "--out", default="unsloth.tensor-types")
    ap.add_argument("--keep-floats", action="store_true")
    args = ap.parse_args()
    pairs = collect(args.src)
    seen = {}
    for name, typ in pairs:
        if name in seen and seen[name] != typ:
            print(f"warn: {name} {seen[name]} vs {typ}", file=sys.stderr)
        seen[name] = typ
    lines = []
    hist = Counter()
    for name, typ in sorted(seen.items()):
        hist[typ] += 1
        if typ in SKIP_TYPES and not args.keep_floats:
            continue
        if name.startswith("mm.") or name.startswith("v."):
            continue
        lines.append(f"{name}={typ}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}  {len(lines)} overrides  {len(seen)} tensors")
    for t, n in hist.most_common():
        print(f"  {t:10} {n}")

if __name__ == "__main__":
    main()
