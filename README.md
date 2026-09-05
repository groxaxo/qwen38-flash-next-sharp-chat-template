# Qwen3.8-Flash-Next Sharp Chat Template

Drop-in Jinja chat template for `Qwen/Qwen3.8-Flash-Next` and abliterated forks (`orcarouter/Qwen3.8-Flash-Next-Uncensored`, etc.), plus GGUF recipes for that checkpoint.

Lineage: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) `v22.5` + [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) terseness block + this repo's `v22.5.2` payload-shape truncate guard.

Version string: `qwen3.8-flash-next-sharp-v22.5.2`

## Files

| file | what |
|---|---|
| [`chat_template.jinja`](./chat_template.jinja) | Sharp + froggeric v22.5 + v22.5.2 guard |
| [`quantize_orcarouter_ad427.sh`](./quantize_orcarouter_ad427.sh) | AtomicChat AD-4.27bpw on OrcaRouter uncensored |
| [`quantize_orcarouter_unsloth_ud.sh`](./quantize_orcarouter_unsloth_ud.sh) | Unsloth UD-Q4_K_XL / UD-IQ4_XS *map* on OrcaRouter uncensored |
| [`extract_gguf_tensor_map.py`](./extract_gguf_tensor_map.py) | header-only dump of `--tensor-type-file` from a GGUF / HF repo |
| [`repro_json_truncation.py`](./repro_json_truncation.py) | jinja2-only repro for the v22.5 xml-mode JSON truncate bug |
| [`ISSUE-froggeric-json-truncation.md`](./ISSUE-froggeric-json-truncation.md) | report to paste on [froggeric HF discussions](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/discussions/new) |

## vs official Flash-Next template

- Official default: `reasoning_effort=xhigh`, thinking on
- This default: `medium` (no extra steering tokens), thinking on, terse system block appended
- Fast path (`enable_thinking=false`): no leftover after-thinking contradictions
- `video_url` content items
- Tool-response truncate guard is payload-shaped (not gated on request `tool_call_format`)
- LM Studio `[TOOL_REQUEST]` stand-down
- Dual XML / JSON tool format
- Default `medium` matches froggeric #72
- No `stock_tool_prompt` hatch

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

OrcaRouter GGUF `Q4_K_M` (~110–119 GB) is stock K-quant. `ffn_down_exps` (ncols=640) and PLE (ncols=160) cannot K-quant and silently fall back.

AtomicChat AD-4.27bpw (~93 GB, ~55 GB resident) is the mixed “Q4.5”.

```bash
SRC=orcarouter/Qwen3.8-Flash-Next-Uncensored LLAMA_CPP=./llama.cpp ./quantize_orcarouter_ad427.sh
```

Traps: keep mmap on; no `--override-tensor` for PLE; `-fit off`; `mxfp4` on `ffn_down_exps` drops the imatrix. Split at 2G so shard 2 is PLE-only.

## Unsloth Dynamic map on OrcaRouter uncensored

Unsloth does not publish an uncensored GGUF. This clones their **per-tensor type table** onto OrcaRouter BF16/FP8. Weights stay OrcaRouter; only the layout is Unsloth Dynamic 3.0.

Does not download the 93–111 GB Unsloth weights. `extract_gguf_tensor_map.py` range-reads GGUF headers and writes `name=TYPE` lines for `llama-quantize --tensor-type-file`.

```bash
chmod +x quantize_orcarouter_unsloth_ud.sh extract_gguf_tensor_map.py
QUANT=UD-Q4_K_XL LLAMA_CPP=./llama.cpp ./quantize_orcarouter_unsloth_ud.sh
QUANT=UD-IQ4_XS ./quantize_orcarouter_unsloth_ud.sh
MAP_ONLY=1 QUANT=UD-Q4_K_XL ./quantize_orcarouter_unsloth_ud.sh
```

Do **not** pass `--token-embedding-type`: on qwen4exp it steals `per_layer_token_embd`. The map names PLE and `token_embd` explicitly. Base type is `q8_0` for anything the map missed.

Imatrix: AtomicChat fallback. Set `RECOMPUTE_IMATRIX=1 CALIB=calib.txt` to rebuild on the uncensored master.

## Changelog

### v22.5.2

Upstream v22.5 gated the JSON-payload truncate skip on `_tool_format == 'json'`. Flash-Next xml-mode never fires that guard. Fix: payload-shape `.startswith('{')` / `.startswith('[')`.

### v22.5.1

Rebase Sharp onto froggeric v22.5. Keep Sharp terseness and `medium` default.

## Sampling

Thinking: `temp=1.0 top_p=0.95 top_k=20`

Instruct: `temp=0.7 top_p=0.8 presence_penalty=1.5 top_k=20`

## License

Apache-2.0, matching froggeric / Sharp.
