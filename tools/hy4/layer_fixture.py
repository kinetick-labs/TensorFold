#!/usr/bin/env python3
"""Tiny Hy4-flash decoder layers through transformers' HYV4FlashDecoderLayer (PR #50903), saved for the CPU
reference in zig/src/families/hy4/layer_ref.zig: an indexed layer whose context outgrows a shrunken index_topk (so
the compressed indexer picks groups) a sliding layer with a 4-token window, both with the dense MLP, and a sliding layer with the latent MoE, on 3D
positions that include a picture span. usage: layer_fixture.py BASE_CONFIG.json OUT.safetensors [ENGRAM_OUT.safetensors]
With ENGRAM_OUT, also a sliding dense layer carrying an Engram memory (config id 2), with its token ids, multipliers
and the n-gram windows the model builds from the time positions."""
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
    hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
    swa_num_attention_heads=3, q_lora_rank=16, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=12,
    v_head_dim=8, swa_qk_nope_head_dim=8, swa_qk_rope_head_dim=12, swa_v_head_dim=8, swa_qk_head_dim=20,
    qk_head_dim=20, head_dim=12, index_n_heads=2, index_head_dim=16, index_topk=8, index_kpool=4,
    compressed_indexer_num_attention_heads=2, compressed_indexer_head_dim=16, compressed_indexer_candidate_topk=8,
    compressed_indexer_compress_ratio=4, compressed_indexer_group_topk=2,
    sliding_window=4, window_size=[3, 0], layer_types=["deepseek_sparse_attention", "sliding_attention", "sliding_attention"],
    mlp_layer_types=["dense", "dense", "sparse"], swa_pattern=[0, 1, 1], engram_layer_ids=None, rope_parameters=rope,
    n_routed_experts=8, num_experts_per_tok=3, moe_intermediate_size=12, routed_expert_hidden_size=16,
    first_k_dense_replace=2, indexer_types=["full", "full", "full"],
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
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.e_score_correction_bias.copy_(torch.randn(cfg.n_routed_experts) * 0.2)
            layer.mlp.experts.gate_up_proj.mul_(4.0)  # push some gates and ups past the swiglu clamp
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

if len(sys.argv) > 3:
    e = dict(small)
    e.update(engram_layer_ids=[2], engram_max_ngram_size=3, engram_n_head_per_ngram=2, engram_n_embed_per_ngram=16,
             engram_layer_head_vocab_sizes={"2": [[31, 37], [41, 43]]}, engram_lookup_table_size=50, engram_pad_id=49,
             engram_fp8_embedding=True)
    ecfg = HYV4FlashTextConfig(**e)
    ecfg._attn_implementation = "eager"
    torch.manual_seed(11)
    layer = m.HYV4FlashDecoderLayer(ecfg, 1).float()
    eng = layer.engram_layer
    with torch.no_grad():
        for name, p in layer.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if p.dim() > 1 else 0.5) + (1.0 if name.endswith("norm.weight") or name.endswith("layernorm.weight") or ".norm_key." in name or ".norm_query." in name or "lookup_per_head_norms" in name else 0.0))
        eng.lookup_table.copy_(torch.randint(0, 40, (50,), dtype=torch.int32))
        eng.hash_multipliers.copy_(eng.draw_hash_multipliers(int(eng.lookup_table.max()) + 1))
    # a picture's pad tokens (masked), ids past the lookup table (read as the pad), a second document from token 10
    ids = torch.tensor([[3, 17, 5, 120120, 120120, 120120, 120120, 120120, 120120, 55, 0, 9, 9, 61, 49, 22]])
    epos = pos.clone()
    for i in range(10, T):
        epos[:, 0, i] = i - 10
    ngram = m.HYV4FlashTextModel.build_ngram_tokens(type("M", (), {"config": ecfg})(), ids, epos)
    emask = (ids != 120120) & (ids != 120124) & (ids != ecfg.engram_pad_id)
    x = torch.randn(1, T, ecfg.hc_mult, ecfg.hidden_size)
    i = torch.arange(T)
    allowed = (i[None, :] <= i[:, None]) & ((i[:, None] - i[None, :]) < ecfg.sliding_window)
    mask = torch.zeros(1, 1, T, T).masked_fill(~allowed, torch.finfo(torch.float32).min)
    cos, sin = m.HYV4FlashRotaryEmbedding(ecfg)(x, epos, layer_type="sliding_attention")
    with torch.no_grad():
        y = layer(x, attention_mask=mask, position_ids=epos, position_embeddings=(cos, sin), ngram_tokens=ngram, engram_mask=emask)
        engram_only = eng(x, ngram, emask)
    eo = {"positions": epos[:, 0].to(torch.int32).contiguous(), "ids": ids[0].to(torch.int32).contiguous(),
          "ngram": ngram[0].to(torch.int32).contiguous(), "multipliers": eng.hash_multipliers.clone(),
          "in": x[0].contiguous(), "out": y[0].contiguous(), "engram_out": engram_only[0].contiguous()}
    for name, p in layer.state_dict().items():
        eo[name] = p.float().contiguous() if p.dtype.is_floating_point else p.contiguous()
    print("engram out norm", float(engram_only.norm()), file=sys.stderr)
    save_file(eo, sys.argv[3], metadata={"source": "transformers HYV4FlashDecoderLayer + HYV4FlashEngramLayer @ b865850d, seed 11"})
json.dump({k: v for k, v in cfg.to_dict().items() if not isinstance(v, (dict,)) or k == "rope_parameters"}, sys.stderr, default=str) if False else None
