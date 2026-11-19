#!/usr/bin/env python3
"""Tiny Hy4-flash decoder layers through transformers' HYV4FlashDecoderLayer (PR #50903), saved for the CPU
reference in zig/src/families/hy4/layer_ref.zig: an indexed layer whose context outgrows a shrunken index_topk (so
the compressed indexer picks groups) and a sliding layer with a 4-token window, both with the dense MLP, on 3D
positions that include a picture span. usage: layer_fixture.py BASE_CONFIG.json OUT.safetensors"""
import json, sys
import torch
from safetensors.torch import save_file
from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as m

base = json.load(open(sys.argv[1]))
base = base.get("text_config", base)
rope = {k: dict(v) for k, v in base["rope_parameters"].items()}
for v in rope.values():
    v["mrope_section"] = [2, 2, 2]
small = dict(base)
small.update(
    hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
    swa_num_attention_heads=3, q_lora_rank=16, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=12,
    v_head_dim=8, swa_qk_nope_head_dim=8, swa_qk_rope_head_dim=12, swa_v_head_dim=8, swa_qk_head_dim=20,
    qk_head_dim=20, head_dim=12, index_n_heads=2, index_head_dim=16, index_topk=8, index_kpool=4,
    compressed_indexer_num_attention_heads=2, compressed_indexer_head_dim=16, compressed_indexer_candidate_topk=8,
    compressed_indexer_compress_ratio=4, compressed_indexer_group_topk=2,
    sliding_window=4, window_size=[3, 0], layer_types=["deepseek_sparse_attention", "sliding_attention"],
    mlp_layer_types=["dense", "dense"], swa_pattern=[0, 1], engram_layer_ids=None, rope_parameters=rope,
    n_routed_experts=4, num_experts_per_tok=2, first_k_dense_replace=2, indexer_types=["full", "full"],
)
small = {k: v for k, v in small.items() if k not in ("architectures", "engram_layer_head_vocab_sizes", "model_type")}
cfg = HYV4FlashTextConfig(**small)
cfg._attn_implementation = "eager"
torch.manual_seed(7)
T = 16
# text 0..2, a 2 x 3 picture at 3..8 (t = 3, h = 3 + r, w = 3 + c), text resumes at 3 + max(2, 3) = 6
pos = torch.zeros(3, 1, T, dtype=torch.long)
for i in range(3):
    pos[:, 0, i] = i
for r in range(2):
    for c in range(3):
        pos[:, 0, 3 + r * 3 + c] = torch.tensor([3, 3 + r, 3 + c])
for i in range(9, T):
    pos[:, 0, i] = 6 + (i - 9)
out = {"positions": pos[:, 0].to(torch.int32).contiguous()}
rot = m.HYV4FlashRotaryEmbedding(cfg)
for idx, kind in enumerate(cfg.layer_types):
    layer = m.HYV4FlashDecoderLayer(cfg, idx).float()
    with torch.no_grad():
        for name, p in layer.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 0.5) + (1.0 if name.endswith("norm.weight") or name.endswith("layernorm.weight") else 0.0))
    x = torch.randn(1, T, cfg.hc_mult, cfg.hidden_size)
    i = torch.arange(T)
    allowed = i[None, :] <= i[:, None]
    if kind == "sliding_attention":
        allowed &= (i[:, None] - i[None, :]) < cfg.sliding_window
    mask = torch.zeros(1, 1, T, T).masked_fill(~allowed, torch.finfo(torch.float32).min)
    cos, sin = rot(x, pos, layer_type=kind)
    with torch.no_grad():
        y = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))
    pre = f"l{idx}."
    out[pre + "in"] = x[0].contiguous()
    out[pre + "out"] = y[0].contiguous()
    for name, p in layer.state_dict().items():
        out[pre + name] = p.float().contiguous()
    print(kind, "out norm", float(y.norm()), file=sys.stderr)
save_file(out, sys.argv[2], metadata={"source": "transformers HYV4FlashDecoderLayer @ b865850d, seed 7"})
json.dump({k: v for k, v in cfg.to_dict().items() if not isinstance(v, (dict,)) or k == "rope_parameters"}, sys.stderr, default=str) if False else None
