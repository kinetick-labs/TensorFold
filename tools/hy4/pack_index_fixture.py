#!/usr/bin/env python3
"""A slice of the SAGE REXL3 pack's model.safetensors.index.json for zig/src/families/hy4 tests: decoder layers 0-2,
experts 0-1, draft layer 0, vision layer 0 and everything global. usage: pack_index_fixture.py INDEX.json OUT.json"""
import json, re, sys

idx = json.load(open(sys.argv[1]))


def keep(k):
    if m := re.match(r"model\.layers\.(\d+)\.", k):
        if int(m[1]) > 2:
            return False
        if e := re.search(r"\.experts\.(\d+)\.", k):
            return int(e[1]) < 2
        return True
    if m := re.match(r"model\.dfly_layers\.(\d+)\.", k):
        return int(m[1]) < 1
    if m := re.match(r"vit\.layers\.(\d+)\.", k):
        return int(m[1]) < 1
    return True


wm = {k: v for k, v in idx["weight_map"].items() if keep(k)}
json.dump({"metadata": idx["metadata"], "weight_map": wm}, open(sys.argv[2], "w"), separators=(",", ":"), sort_keys=True)
print(len(wm), "tensors", file=sys.stderr)
