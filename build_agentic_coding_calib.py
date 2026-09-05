#!/usr/bin/env python3
"""Build a high-quality agentic-coding imatrix corpus for Qwen3.8-Flash-Next.
See repo README / agent_calib_agentic_coding_3x3090.json.
"""
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

def approx_tokens(s):
    return max(1, len(s) // 4)

def wrap_turn(role, body):
    return f"{IM_START}{role}\n{body.strip()}\n{IM_END}\n"

def xml_tool_call(name, args):
    parts = [f"<tool_call>\n<function={name}>"]
    for k, v in args.items():
        val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        parts.append(f"<parameter={k}>{val}</parameter>")
    parts.append("</function>\n</tool_call>")
    return "\n".join(parts)

def xml_tool_response(body):
    return f"<tool_response>\n{body.strip()}\n</tool_response>"

def hf_load(name, split=None, config=None):
    from datasets import load_dataset
    kwargs = {}
    if split:
        kwargs["split"] = split
    ds = load_dataset(name, config, **kwargs) if config else load_dataset(name, **kwargs)
    if hasattr(ds, "keys") and split is None:
        ds = ds[next(iter(ds.keys()))]
    return ds

def take_swe_verified(n, rng):
    out = []
    try:
        rows = list(hf_load("princeton-nlp/SWE-bench_Verified", split="test"))
    except Exception as e:
        print(f"warn: SWE-bench_Verified failed ({e})", file=sys.stderr)
        return out
    rng.shuffle(rows)
    for row in rows:
        if len(out) >= n:
            break
        issue = (row.get("problem_statement") or "").strip()
        patch = (row.get("patch") or row.get("gold_patch") or "").strip()
        if len(issue) < 80 or len(patch) < 40:
            continue
        fails = row.get("FAIL_TO_PASS") or ""
        if isinstance(fails, list):
            fails = "\n".join(fails)
        user = f"Repo {row.get('repo','')} ({row.get('instance_id','')}). Fix this GitHub issue.\n\n{issue[:6000]}\n\nFailing tests:\n{str(fails)[:1500]}"
        assistant = "<think>\nInspect tests, locate files, apply a minimal patch, re-run FAIL_TO_PASS.\n</think>\n" + xml_tool_call("execute_bash", {"command": patch[:8000]})
        out.append(wrap_turn("user", user) + wrap_turn("assistant", assistant) + wrap_turn("tool", xml_tool_response(patch[:8000])) + wrap_turn("assistant", patch[:4000]))
    print(f"swe_verified {len(out)}", file=sys.stderr)
    return out

def rewrite_tools(tools):
    if isinstance(tools, dict):
        tools = [tools]
    chunks = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or t
        name = fn.get("name") or t.get("name") or "execute_bash"
        args = fn.get("arguments") or t.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"input": str(args)}
        if name in ("run", "bash", "shell", "terminal", "execute_bash", "run_command"):
            name = "execute_bash"
            if "command" not in args:
                args = {"command": args.get("cmd") or args.get("input") or json.dumps(args)}
        chunks.append(xml_tool_call(name, args))
    return "\n".join(chunks)

def flatten_messages(msgs):
    if not msgs:
        return ""
    if isinstance(msgs, str):
        return wrap_turn("user", msgs[:8000])
    if isinstance(msgs, dict):
        msgs = msgs.get("messages") or msgs.get("turns") or [msgs]
    buf = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or m.get("from") or "user").lower()
        content = m.get("content") or m.get("value") or ""
        if isinstance(content, list):
            content = "\n".join(str(p.get("text") or p.get("content") or p) if isinstance(p, dict) else str(p) for p in content)
        tools = m.get("tool_calls") or m.get("function_call")
        if tools:
            content = str(content) + "\n" + rewrite_tools(tools)
        if role == "system":
            continue
        if role not in ("user", "assistant", "tool"):
            role = "assistant" if "assist" in role else "user"
        if str(content).strip():
            buf.append(wrap_turn(role, str(content)[:12000]))
        if len(buf) >= 16:
            break
    return "".join(buf)

def take_openhands(n, rng):
    out, rows = [], []
    for name, split in (("SWE-Gym/OpenHands-SFT-Trajectories", "train"), ("nebius/SWE-rebench-openhands-trajectories", None)):
        try:
            rows.extend(list(hf_load(name, split=split)))
            if len(rows) > 4000:
                break
        except Exception as e:
            print(f"warn: {name} failed ({e})", file=sys.stderr)
    rng.shuffle(rows)
    for row in rows:
        if len(out) >= n:
            break
        text = flatten_messages(row.get("messages") or row.get("trajectory") or row.get("conversations"))
        if text and approx_tokens(text) > 200:
            out.append(text[:24000])
    print(f"openhands {len(out)}", file=sys.stderr)
    return out

def take_eaddario(kind, n_chars):
    try:
        from huggingface_hub import hf_hub_download
        raw = Path(hf_hub_download("eaddario/imatrix-calibration", filename=f"{kind}.txt")).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"warn: eaddario {kind} failed ({e})", file=sys.stderr)
        return []
    buf, total = [], 0
    for para in re.split(r"\n{2,}", raw):
        para = para.strip()
        if len(para) < 80:
            continue
        role = "user" if kind.startswith("tools") else "assistant"
        chunk = wrap_turn(role, para[:4000])
        buf.append(chunk)
        total += len(chunk)
        if total >= n_chars:
            break
    print(f"eaddario {kind} {len(buf)} blocks", file=sys.stderr)
    return buf

def take_atomic_agentic(n_chars):
    try:
        from huggingface_hub import hf_hub_download
        for fname in ("eval/agentic/eval_agentic.txt", "builds/agentic/calib_train.txt", "agentic.txt"):
            try:
                raw = Path(hf_hub_download("AtomicChat/calib-corpora", filename=fname)).read_text(encoding="utf-8", errors="replace")
                print(f"atomic {fname} {len(raw)} chars", file=sys.stderr)
                return [raw[:n_chars]]
            except Exception:
                continue
    except Exception as e:
        print(f"warn: AtomicChat/calib-corpora ({e})", file=sys.stderr)
    return []

def pad_eot(text, ctx):
    need = ctx * 4
    if len(text) % need == 0:
        return text
    return text + ("<|endoftext|>" * ((need - (len(text) % need)) // 13 + 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="calib/calib_agentic_coding.txt")
    ap.add_argument("--seed", type=int, default=38)
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--swe-n", type=int, default=400)
    ap.add_argument("--oh-n", type=int, default=250)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    parts = take_swe_verified(args.swe_n, rng) + take_openhands(args.oh_n, rng) + take_atomic_agentic(400000) + take_eaddario("tools_small", 350000) + take_eaddario("code_small", 700000)
    rng.shuffle(parts)
    out, toks = [], 0
    for p in parts:
        t = approx_tokens(p)
        if toks + t > args.max_tokens:
            break
        out.append(p.rstrip() + "\n")
        toks += t
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pad_eot("\n".join(out), 2048)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} chars={len(text)} approx_tokens={len(text)//4} docs={len(out)}")
    if len(text) < 50000:
        print("error: corpus too small — check HF auth", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
