#!/usr/bin/env python3
"""Repro: max_tool_response_chars corrupts JSON tool responses in xml mode (v22.5).
Usage: python3 repro_json_truncation.py /path/to/chat_template.jinja"""
import json, sys
from jinja2 import Environment, ChainableUndefined
from jinja2.exceptions import TemplateError

E = Environment(trim_blocks=True, lstrip_blocks=True, undefined=ChainableUndefined)
E.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(TemplateError(m))
tpl = E.from_string(open(sys.argv[1], encoding="utf-8").read())

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
PAYLOAD = json.dumps({"rows": [{"id": i} for i in range(40)]})
MSGS = [{"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}]},
        {"role": "tool", "content": PAYLOAD}]

for fmt in ("xml", "json"):
    out = tpl.render(messages=MSGS, tools=TOOLS, add_generation_prompt=True,
                     tool_call_format=fmt, max_tool_response_chars=100)
    body = out.split("<tool_response>\n")[1].split("\n[TRUNCATED")[0].split("\n</tool_response>")[0]
    try:
        json.loads(body); verdict = "OK  - payload intact"
    except json.JSONDecodeError:
        verdict = "BUG - payload truncated into invalid JSON"
    print(f"tool_call_format={fmt:<5} {verdict}")
