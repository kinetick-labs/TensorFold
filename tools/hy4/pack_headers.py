#!/usr/bin/env python3
"""Every tensor's shard, dtype and shape from a pack's shard headers (no tensor data), both indexes, as one JSON
{"name": ["shard", "DTYPE", [shape]]} for zig/src/families/hy4/load_plan.zig. usage: pack_headers.py PACK_DIR OUT.json
With --fixture, keeps the slice load_plan's test plans (layers 0-2, experts 0-7, the vision tower, the globals and
layer 1's Engram table) with the router shrunk to 8 experts."""
import json, os, re, struct, sys
d, out_path = sys.argv[1], sys.argv[2]
out = {}
for idx in ("model.safetensors.index.json", "engram.safetensors.index.json"):
    wm = json.load(open(os.path.join(d, idx)))["weight_map"]
    for s in sorted(set(wm.values())):
        f = open(os.path.join(d, s), "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
        for k, v in h.items():
            if k != "__metadata__": out[k] = [s, v["dtype"], v["shape"]]
if "--fixture" in sys.argv:
    keep = {}
    for k, v in out.items():
        m = re.match(r"model\.layers\.(\d+)\.", k)
        if m and int(m.group(1)) > 2: continue
        e = re.search(r"\.experts\.(\d+)\.", k)
        if e and int(e.group(1)) > 7: continue
        if k.startswith("model.dfly_layers."): continue
        if k.endswith("mlp.gate.weight"): v = [v[0], v[1], [8, v[2][1]]]
        if k.endswith("mlp.gate.e_score_correction_bias"): v = [v[0], v[1], [8]]
        keep[k] = v
    out = keep
json.dump(out, open(out_path, "w"), separators=(",", ":"))
print(len(out))
