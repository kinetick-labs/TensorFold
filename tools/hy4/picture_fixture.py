#!/usr/bin/env python3
"""Hy4-flash picture preprocessing reference: transformers' HYV4VLImageProcessor with the release's
processor_config.json on procedurally drawn pictures (the same formula as zig/src/families/hy4/root.zig), sampled
every 97th value of pixel_values so the fixture stays small. usage: picture_fixture.py PROCESSOR_CONFIG.json OUT.json"""
import json, sys
import numpy as np
from PIL import Image
from transformers.models.hy_v4_vl.image_processing_hy_v4_vl import HYV4VLImageProcessor

cfg = json.load(open(sys.argv[1]))["image_processor"]
proc = HYV4VLImageProcessor(**{k: v for k, v in cfg.items() if k not in ("image_processor_type", "device")})


def draw(w, h):
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x * 7 + y * 13 + c * 50 + (x * y) % 31) % 256 for c in range(3)], axis=-1).astype(np.uint8)


out = []
for w, h in [(53, 37), (300, 200), (75, 1500)]:
    r = proc(images=Image.fromarray(draw(w, h)), return_tensors="np")
    pv = r["pixel_values"].astype(np.float64)
    out.append({"width": w, "height": h, "grid": r["image_grid_thw"][0].tolist(), "shape": list(pv.shape),
                "sum": float(pv.sum()), "abs_sum": float(np.abs(pv).sum()), "every97": pv.reshape(-1)[::97].tolist()})
    print(w, h, out[-1]["grid"], out[-1]["shape"], file=sys.stderr)
json.dump(out, open(sys.argv[2], "w"))
