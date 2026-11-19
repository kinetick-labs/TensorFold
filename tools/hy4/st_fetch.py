#!/usr/bin/env python3
"""Fetch named tensors out of a public Hugging Face safetensors checkpoint with HTTP range reads
(no whole-shard download). usage: st_fetch.py REPO REV OUT.safetensors REGEX"""
import json, re, struct, sys, urllib.request
repo, rev, out, pat = sys.argv[1:5]
base = f"https://huggingface.co/{repo}/resolve/{rev}/"
def get(url, a=None, b=None):
    req = urllib.request.Request(url)
    if a is not None: req.add_header("Range", f"bytes={a}-{b}")
    return urllib.request.urlopen(req).read()
wm = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
want = sorted(k for k in wm if re.search(pat, k))
by = {}
for k in want: by.setdefault(wm[k], []).append(k)
tensors = {}
for shard, keys in sorted(by.items()):
    n = struct.unpack("<Q", get(base + shard, 0, 7))[0]
    hdr = json.loads(get(base + shard, 8, 8 + n - 1))
    for k in keys:
        m = hdr[k]; s, e = m["data_offsets"]
        tensors[k] = (m["dtype"], m["shape"], get(base + shard, 8 + n + s, 8 + n + e - 1))
        print(k, m["dtype"], m["shape"], file=sys.stderr)
meta, off = {}, 0
for k in sorted(tensors):
    d, sh, b = tensors[k]; meta[k] = {"dtype": d, "shape": sh, "data_offsets": [off, off + len(b)]}; off += len(b)
meta["__metadata__"] = {"source": f"{repo}@{rev}"}
h = json.dumps(meta, separators=(",", ":")).encode(); h += b" " * ((8 - len(h) % 8) % 8)
with open(out, "wb") as f:
    f.write(struct.pack("<Q", len(h))); f.write(h)
    for k in sorted(tensors): f.write(tensors[k][2])
