#!/usr/bin/env python3
"""The Hy4 vision tower (transformers' HYV4VLVisionModel, which HYV4FlashModel builds from `vision_config`) run on
fixed pixels, saved for the CPU reference in zig/src/families/hy4/vision_ref.zig. Weights keep the checkpoint's names
(`vit.layers.N.input_layernorm`, `mlp.dense_h_to_4h`, `vit.perceive.proj.0`...), which transformers renames on load.
usage:
  vision_fixture.py tiny BASE_CONFIG.json OUT.safetensors
      a shrunken random tower on two pictures at once (grids 4 x 6 and 6 x 4, so attention stays per picture);
      the weights, pixels, grids and merged features are all in OUT
  vision_fixture.py real BASE_CONFIG.json VIT.safetensors OUT.safetensors
      the release's tower (VIT holds the pack's `vit.*` tensors) on one 4 x 6 grid of fixed pixels, all 27 layers;
      OUT holds only the pixels, the grid and the merged features (the weights stay in the pack)"""
import json, re, sys
import torch
from safetensors.torch import save_file, load_file
from transformers.models.hy_v4_vl.configuration_hy_v4_vl import HYV4VLVisionConfig
from transformers.models.hy_v4_vl.modeling_hy_v4_vl import HYV4VLVisionModel

mode = sys.argv[1]
base = json.load(open(sys.argv[2]))["vision_config"]
base = {k: v for k, v in base.items() if k in HYV4VLVisionConfig.__dataclass_fields__ or k in ("hidden_act",)}

def to_hf(name):
    # the hy_v4_vl checkpoint renames (conversion_mapping.py), minus the module placement
    n = re.sub(r"^vit\.", "", name)
    n = n.replace("mlp.dense_h_to_4h", "mlp.fc1").replace("mlp.dense_4h_to_h", "mlp.fc2")
    n = re.sub(r"^layers\.(\d+)\.input_layernorm\.", r"layers.\1.layer_norm1.", n)
    n = re.sub(r"^layers\.(\d+)\.post_attention_layernorm\.", r"layers.\1.layer_norm2.", n)
    n = n.replace("perceive.norm.", "perceive.before_rms.").replace("perceive.proj.0.", "perceive.proj_conv.").replace("perceive.proj.2.", "perceive.proj_out.")
    return n.replace("perceive.", "patch_merger.")

def from_hf(name):
    n = name.replace("patch_merger.", "perceive.").replace("perceive.before_rms.", "perceive.norm.")
    n = n.replace("perceive.proj_conv.", "perceive.proj.0.").replace("perceive.proj_out.", "perceive.proj.2.")
    n = n.replace("mlp.fc1", "mlp.dense_h_to_4h").replace("mlp.fc2", "mlp.dense_4h_to_h")
    n = re.sub(r"layer_norm1\.", "input_layernorm.", n)
    n = re.sub(r"layer_norm2\.", "post_attention_layernorm.", n)
    return "vit." + n

torch.manual_seed(0)
if mode == "tiny":
    cfg = HYV4VLVisionConfig(**{**base, "hidden_size": 32, "intermediate_size": 48, "num_attention_heads": 4, "num_hidden_layers": 2,
                                "patch_size": 4, "out_hidden_size": 24, "max_image_size": 32})
    model = HYV4VLVisionModel(cfg).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    grids = [[1, 4, 6], [1, 6, 4]]
    out_path = sys.argv[3]
else:
    cfg = HYV4VLVisionConfig(**base)
    model = HYV4VLVisionModel(cfg).float().eval()
    sd = {to_hf(k): v.float() for k, v in load_file(sys.argv[3]).items()}
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    grids = [[1, 4, 6]]
    out_path = sys.argv[4]
cfg._attn_implementation = "eager"
p = cfg.patch_size
n = sum(t * h * w for t, h, w in grids)
pixels = (torch.rand(n, 3 * p * p) * 2 - 1).float()
grid = torch.tensor(grids, dtype=torch.long)
with torch.no_grad():
    out = model(pixels, grid_thw=grid)
feats = out.pooler_output.squeeze(0).float().contiguous()
save = {"pixels": pixels, "grid": grid.to(torch.int32), "features": feats}
if mode == "tiny":
    save.update({from_hf(k): v.float().contiguous() for k, v in model.state_dict().items()})
save_file(save, out_path)
print(mode, feats.shape, float(feats.abs().mean()))
