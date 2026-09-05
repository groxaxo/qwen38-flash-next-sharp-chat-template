# Qwen3.8-Flash-Next Sharp Chat Template

Drop-in Jinja chat template plus GGUF recipes for `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8`.

Lineage: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) `v22.5` + [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) + this repo `v22.5.2` payload-shape truncate guard.

Version string: `qwen3.8-flash-next-sharp-v22.5.2`

## Files

| file | what |
|---|---|
| [`chat_template.jinja`](./chat_template.jinja) | Sharp + froggeric v22.5 + v22.5.2 guard |
| [`quantize_orcarouter_fp8.sh`](./quantize_orcarouter_fp8.sh) | **main** — FP8 → Q8 master → Unsloth map and/or AD-4.27 |
| [`extract_gguf_tensor_map.py`](./extract_gguf_tensor_map.py) | header-only `--tensor-type-file` dump |
| [`quantize_orcarouter_unsloth_ud.sh`](./quantize_orcarouter_unsloth_ud.sh) | Unsloth map only |
| [`quantize_orcarouter_ad427.sh`](./quantize_orcarouter_ad427.sh) | AD-4.27 only |
| [`repro_json_truncation.py`](./repro_json_truncation.py) | xml-mode JSON truncate repro |
| [`ISSUE-froggeric-json-truncation.md`](./ISSUE-froggeric-json-truncation.md) | report for froggeric HF |

## Quantize from Uncensored-FP8

Base: `orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8` (~186 GB, **gated**). Converter writes Q8_0 with `--fp8-as-q8`, then applies Unsloth Dynamic types and/or AtomicChat AD-4.27. Does not download Unsloth weights — only GGUF headers for the type map.

```bash
hf auth login   # accept the FP8 repo terms in the browser first
chmod +x quantize_orcarouter_fp8.sh extract_gguf_tensor_map.py

RECIPE=unsloth QUANT=UD-Q4_K_XL LLAMA_CPP=./llama.cpp ./quantize_orcarouter_fp8.sh
RECIPE=unsloth QUANT=UD-IQ4_XS ./quantize_orcarouter_fp8.sh
RECIPE=ad427 ./quantize_orcarouter_fp8.sh
RECIPE=unsloth+ad427 QUANT=UD-Q4_K_XL ./quantize_orcarouter_fp8.sh
SRC=/data/Qwen3.8-Flash-Next-Uncensored-FP8 RECIPE=unsloth ./quantize_orcarouter_fp8.sh
```

Disk: FP8 ~186 GB + Q8 master ~176 GB + output ~94–111 GB.
Do **not** pass `--token-embedding-type` (steals PLE). Split 2G isolates PLE. `-fit off`, mmap on, no `--override-tensor` for PLE.

## Template

```bash
llama-server -m MODEL.gguf --jinja --chat-template-file chat_template.jinja \
  --reasoning-format deepseek
```

Default `reasoning_effort=medium`, `terse=true`, `tool_call_format=xml`.

## License

Apache-2.0.
