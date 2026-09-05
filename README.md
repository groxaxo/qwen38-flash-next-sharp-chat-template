# Qwen3.8-Flash-Next Sharp Chat Template

Drop-in Jinja chat template for `Qwen/Qwen3.8-Flash-Next` and abliterated forks (`orcarouter/Qwen3.8-Flash-Next-Uncensored`, etc.), plus the AtomicChat AD-4.27bpw quant recipe for that checkpoint.

Lineage: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) `v22.5` + [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) terseness block + this repo's `v22.5.2` payload-shape truncate guard.

Version string: `qwen3.8-flash-next-sharp-v22.5.2`

## Files

| file | what |
|---|---|
| [`chat_template.jinja`](./chat_template.jinja) | Sharp + froggeric v22.5 + v22.5.2 guard |
| [`quantize_orcarouter_ad427.sh`](./quantize_orcarouter_ad427.sh) | AD-4.27bpw (“Q4.5”) quant of OrcaRouter uncensored |
| [`repro_json_truncation.py`](./repro_json_truncation.py) | jinja2-only repro for the v22.5 xml-mode JSON truncate bug |
| [`ISSUE-froggeric-json-truncation.md`](./ISSUE-froggeric-json-truncation.md) | report to paste on [froggeric HF discussions](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/discussions/new) |

## vs official Flash-Next template

- Official default: `reasoning_effort=xhigh`, thinking on
- This default: `medium` (no extra steering tokens), thinking on, terse system block appended
- Fast path (`enable_thinking=false`): no leftover “after thinking” contradictions
- `video_url` content items
- Tool-response truncate guard is payload-shaped (not gated on request `tool_call_format`)
- LM Studio `[TOOL_REQUEST]` stand-down
- Dual XML / JSON tool format
- Default `medium` matches froggeric #72 (xhigh can burn the token budget and return empty content)
- No `stock_tool_prompt` hatch — non-thinking tool block is already aligned with v22.5

With `terse=false` and `reasoning_effort=xhigh`, non-tool renders are byte-identical to Qwen stock.

## Apply the template

```bash
llama-server -m MODEL.gguf --jinja --chat-template-file chat_template.jinja \
  --reasoning-format deepseek

gguf-new-metadata --chat-template-file chat_template.jinja in.gguf out.gguf
```

LM Studio: drop `chat_template.jinja` next to the model, rescan.

```json
{"chat_template_kwargs": {"terse": true, "reasoning_effort": "medium"}}
```

## kwargs

| key | default | notes |
|---|---|---|
| `terse` | true | `false` = no Sharp system block |
| `enable_thinking` | true | or `reasoning_effort: "none"` / `<\|think_off\|>` |
| `reasoning_effort` | medium | `low` / `medium` / `xhigh` (+ aliases) |
| `preserve_thinking` | true | keep history `<think>` |
| `tool_call_format` | xml | or `json` — Flash-Next speaks XML |
| `suppress_tool_instructions` | auto | stands down if `[TOOL_REQUEST]` present |
| `max_tool_arg_chars` | 0 | 0 = unlimited; truncates XML tool-call args |
| `max_tool_response_chars` | 0 | 0 = unlimited; skips payloads that start with `{` or `[` |
| `add_vision_id` | false | prefixes `Picture N:` / `Video N:` |
| `auto_disable_thinking_with_tools` | false | force thinking off when tools are present |

## Quantize OrcaRouter uncensored to AD-4.27bpw

OrcaRouter GGUF `Q4_K_M` (~110–119 GB) is stock K-quant. `ffn_down_exps` (ncols=640) and PLE (ncols=160) cannot K-quant and silently fall back, so the file stays fat and the PLE sits in a weight shard (Metal wires it).

AtomicChat AD-4.27bpw (~93 GB, ~55 GB resident) is the mixed “Q4.5”:

| group | type | file share | bpw contrib |
|---|---|---|---|
| PLE `per_layer_token_embd` | Q5_1 | 41% | 1.74 |
| `ffn_gate/up_exps` | IQ2_S, IQ3_S on blk 0–3 + 40–47 | 29% | 1.24 |
| `ffn_down_exps` | IQ4_NL | 24% | 1.03 |
| rest | Q8_0 | 5% | 0.23 |

Abliteration only touched 149 residual writers. Routers / PLE / vision are stock, so the AtomicChat imatrix is a usable fallback. Recompute on this checkpoint if you can.

Do **not** start from the MLX-4bit (experts + PLE stay high precision, ~7.85 bpw). Start from BF16 or FP8→Q8.

```bash
# llama.cpp with qwen4exp
chmod +x quantize_orcarouter_ad427.sh
SRC=orcarouter/Qwen3.8-Flash-Next-Uncensored \
LLAMA_CPP=./llama.cpp \
./quantize_orcarouter_ad427.sh
```

FP8 path on 128 GB RAM:

```bash
SRC=/path/to/Qwen3.8-Flash-Next-Uncensored-FP8 OUTTYPE=q8_0 ./quantize_orcarouter_ad427.sh
```

Serve:

```bash
llama-server -m ./out/Qwen38FN-Unc-AD-4.27bpw-00001-of-*.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf \
  -ngl 99 -c 32768 --jinja -fit off --reasoning-format deepseek
```

Traps: keep mmap on; no `--override-tensor` for PLE; `-fit off`; `mxfp4` on `ffn_down_exps` throws away the imatrix. Split at 2G so shard 2 is PLE-only.

## Changelog

### v22.5.2

Upstream v22.5 gated the JSON-payload truncate skip on `_tool_format == 'json'`. Flash-Next (and every xml-mode deploy) never fires that guard, so `max_tool_response_chars` slices JSON tool *responses* into invalid JSON.

Fix: detect payload shape with `.startswith('{')` / `.startswith('[')`. Default path unchanged when `max_tool_response_chars` is unset.

### v22.5.1

Rebase Sharp onto froggeric v22.5 (`video_url`, v22.5 tool-prompt alignment). Keep Sharp terseness and `medium` default.

## Sampling

Thinking: `temp=1.0 top_p=0.95 top_k=20`

Instruct: `temp=0.7 top_p=0.8 presence_penalty=1.5 top_k=20`

## License

Apache-2.0, matching froggeric / Sharp.
