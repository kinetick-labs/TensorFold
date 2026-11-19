#!/usr/bin/env python3
"""Hy4-flash chat template cases and their jinja2 renders, as transformers' apply_chat_template sets jinja up
(sandbox, trim/lstrip blocks, loopcontrols, raise_exception, its tojson). usage: template_parity.py TEMPLATE OUTDIR
Then run zig/src/core/template/hy4_parity_test.zig with HY4_TEMPLATE_DIR=OUTDIR."""
import json, os, shutil, sys
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

src, outdir = sys.argv[1:3]
os.makedirs(outdir, exist_ok=True)
shutil.copy(src, os.path.join(outdir, "chat_template.jinja"))
tools=[{"type":"function","function":{"name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
cases=[
 {"messages":[{"role":"user","content":[{"type":"image"},{"type":"text","text":"What is in this picture?"}]}],"add_generation_prompt":True,"reasoning_effort":"high"},
 {"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":[{"type":"text","text":"Compare"},{"type":"image_url","image_url":{"url":"x"}},{"type":"image"}]}],"add_generation_prompt":True,"reasoning_effort":"no_think"},
 {"messages":[{"role":"user","content":"Weather in Paris?"},{"role":"assistant","content":"","reasoning_content":"Need the tool.","tool_calls":[{"type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}]},{"role":"tool","content":"{\"t\": 12}"},{"role":"assistant","content":"12 degrees."},{"role":"user","content":"Thanks"}],"tools":tools,"add_generation_prompt":True},
 {"messages":[{"role":"user","content":"hi"}],"add_generation_prompt":True,"reasoning_effort":"medium"},
]
json.dump(cases, open(os.path.join(outdir, "cases.json"), "w"))


def raise_exception(m):
    raise jinja2.exceptions.TemplateError(m)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = raise_exception
env.filters["tojson"] = tojson
t = env.from_string(open(src).read())
out = []
for case in cases:
    try:
        out.append({"text": t.render(**case)})
    except Exception as e:
        out.append({"error": str(e)})
json.dump(out, open(os.path.join(outdir, "ref.json"), "w"), ensure_ascii=False, indent=1)
