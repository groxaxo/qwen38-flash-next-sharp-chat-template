# max_tool_response_chars corrupts JSON tool responses in xml mode (v22.5)

`_is_json_payload` is gated on the request wire format rather than the response payload shape:

```jinja
{%- set _is_json_payload = (_tool_format == 'json' and content | trim | length > 0 and (content | trim)[:1] in ('{', '[')) %}
```

`_tool_format` defaults to `xml`, so on every xml-mode deployment the guard never fires and a JSON tool response over the limit gets sliced mid-structure. The reason not to truncate JSON — truncation makes it unparseable — applies identically in both modes.

```
$ python3 repro_json_truncation.py chat_template.jinja
tool_call_format=xml   BUG - payload truncated into invalid JSON
tool_call_format=json  OK  - payload intact
```

The v22.5 regression test for smart JSON response truncation passes because it exercises the json path only, which is why this slipped through.

Fix, dropping the conjunct:

```jinja
{%- set _is_json_payload = ((content | trim).startswith('{') or (content | trim).startswith('[')) %}
```

Renders identically when `max_tool_response_chars` is unset, so the default path is untouched. Secondary benefit: it drops the tuple-`in`, which is the only construct in that line I haven't seen exercised under minja.

Repro (jinja2 only, no model): [`repro_json_truncation.py`](./repro_json_truncation.py)

Patched in: https://github.com/groxaxo/qwen38-flash-next-sharp-chat-template (`v22.5.2`)
