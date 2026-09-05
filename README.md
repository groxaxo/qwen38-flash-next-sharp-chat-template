# Qwen3.8-Flash-Next Sharp Chat Template

Drop-in Jinja chat template for `Qwen/Qwen3.8-Flash-Next` and abliterated forks (OrcaRouter, etc.).

Lineage: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) `v22.5` + [peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates) terseness block + this repo's `v22.5.2` payload-shape truncate guard.

Version string inside the template: `qwen3.8-flash-next-sharp-v22.5.2`

## vs official Flash-Next template

- Official default: `reasoning_effort=xhigh`, thinking on
- This default: `medium` (no extra steering tokens), thinking on, terse system block appended
- Fast path (`enable_thinking=false`): no leftover “after thinking” contradictions
- `video_url` content items
- Tool-response truncate guard is payload-shaped (not gated on request `tool_call_format`)
- LM Studio `[TOOL_REQUEST]` stand-down
- Dual XML / JSON tool format
- Default `medium` matches froggeric #72 (xhigh can burn the token budget and return empty content)
- No `stock_tool_prompt` hatch — non-thinking tool block is already aligned with v22.5. Restore from froggeric if you need to bisect Toolathlon regressions

With `terse=false` and `reasoning_effort=xhigh`, non-tool renders are byte-identical to Qwen stock.

## Apply

```bash
# llama.cpp
llama-server -m MODEL.gguf --jinja --chat-template-file chat_template.jinja \
  --reasoning-format deepseek

# bake into GGUF
gguf-new-metadata --chat-template-file chat_template.jinja in.gguf out.gguf
```

LM Studio: drop `chat_template.jinja` next to the model, rescan.

vLLM / transformers:

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
