#!/usr/bin/env python3
"""Hy4-flash Engram hash reference, as transformers' HYV4FlashEngramLayer computes it (PR #50903, Tencent's
reference): the per-layer odd multipliers and the hash ids of one token window. The multipliers and the
vectors in zig/src/families/hy4/root.zig come from here.

usage: engram_reference.py CONFIG.json LOOKUP_TABLE.npy [LAYER_ID TOKEN ...]
LOOKUP_TABLE.npy is model.layers.N.engram.lookup_table (identical across layers) cut out with st_fetch.py."""
import json, sys
import numpy as np

cfg = json.load(open(sys.argv[1]))
cfg = cfg.get("text_config", cfg)
table = np.load(sys.argv[2]).astype(np.int64)
seed, max_n, per = cfg["engram_seed"], cfg["engram_max_ngram_size"], cfg["engram_n_head_per_ngram"]
cvocab = int(table.max()) + 1  # transformers' _init_weights redraws from the loaded map


def multipliers(layer_id):
    half = max(1, np.iinfo(np.int64).max // cvocab // 2)
    g = np.random.default_rng(seed + 10007 * layer_id)
    return g.integers(0, half, size=max_n, dtype=np.int64) * 2 + 1


for lid in cfg["engram_layer_ids"]:
    print("layer", lid, "multipliers", multipliers(lid).tolist())

if len(sys.argv) > 4:
    lid, toks = int(sys.argv[3]), [int(t) for t in sys.argv[4:]]
    pad = cfg["engram_pad_id"]
    sizes = [s for order in cfg["engram_layer_head_vocab_sizes"][str(lid)] for s in order]
    m = multipliers(lid)
    norm = lambda t: table[t] if t < len(table) else table[pad]
    # window [t, t-1, ..., t-max_n+1]; slots before the document start read as the pad
    out = []
    for at in range(len(toks)):
        w = [norm(toks[at - k]) if at - k >= 0 else norm(pad) for k in range(max_n)]
        mixed = [int(np.int64(w[k]) * m[k]) for k in range(max_n)]
        for n in range(2, max_n + 1):
            x = mixed[0]
            for k in range(1, n):
                x ^= mixed[k]
            mods = sizes[(n - 2) * per:(n - 1) * per]
            out.append([x % q for q in mods])
        print(at, sum(out[-(max_n - 1):], []))
