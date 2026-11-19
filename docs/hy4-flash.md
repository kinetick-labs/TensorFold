# Hy4-flash on TensorFold: gap analysis and plan

Goal: serve `vcruz305/Hy4-Flash-SAGE-REXL3-4.84bpw` (Tencent `tencent/Hy4-flash` quantized with SAGE REXL3 by vcruz305)
with pictures, on one DGX Spark (GB10, 128 GB unified, sm_121).

Sources read: the base model's `config.json`, `processor_config.json`, `chat_template.jinja` and safetensors index
(`tencent/Hy4-flash@b2498eba`), Tencent's modeling in transformers PR #50903 (`hy_v4_flash`, `hy_v4_vl`), the quant
pack's model card. Credits: Tencent Hunyuan (model, reference code), vcruz305 (SAGE REXL3 pack), the transformers
authors of the port. License: tencent-hy-license (see the pack's LICENSE).

## The model

| | |
|---|---|
| params | 220B total (64B of them Engram tables), ~12B active |
| layers | 42; layer 0 dense FFN 13824, layers 1-41 latent MoE |
| MoE | 512 routed experts, top-16 + 1 shared, latent 1536, expert width 1536, sigmoid router + correction bias, routed scale 3.998, SwiGLU clamp 10 |
| hidden / vocab | 3072 / 120832, RMSNorm eps 1e-5 |
| residual | iHC, 4 streams: per sublayer fn [8, 4*3072], base[8], scale[2]; pre = sigmoid + eps, post = 2 sigmoid + eps, no Sinkhorn; hc_head before the final norm |
| attention | gated MLA (q_lora 1024, kv_lora 512, nope 192, rope 64, v 256), sigmoid output gate, learned sinks, post-projection norm |
| layer kinds | 12 indexed layers (0, 1, 5, 9 ... 41): 32 heads, rope theta 1e7, compressed indexer; 30 sliding layers: 64 heads, window 512, theta 1e4 |
| indexer | 16 heads x 128; keys pooled in groups of 4 (fp32 value/gate proj + intra-group APE, softmax, key norm); top-511 groups + own group = 2048 tokens; dense under 2048 |
| rope | M-RoPE interleaved, sections [12, 10, 10] |
| Engram | decoder layers 1, 11, 21, 31; orders 2-3 x 8 heads, 96 dims, prime moduli ~10.3M per head, normalized-vocab hash, FP8 round trip, sqrt-signed sigmoid gate per HC stream |
| draft | DFly2: 5 layers on target layers 25/29/33/37/41, 5 speculative tokens, mask 120009 |
| vision | SigLIP2-So400m-like: 27 layers, 1152, MLP 4304, 16 heads, patch 16, LayerNorm+bias, exact GELU, learned 128x128 position grid bilinear-resampled; merger RMSNorm -> 2x2 conv 1152->4608 -> GELU -> 1x1 -> 3072, plus a newline column |
| pictures | smart resize factor 32, 4096..4194304 px, bicubic, mean/std 0.5; (h/2)*(w/2+1) tokens; ids start 120118 end 120119 pad 120120 newline 120121 |

## The pack

- 12 GPU shards, 99.75 GB (92.9 GiB): REXL3 trellis weights at 4.84 bpw average.
- Engram sidecar, 2 shards, 64.68 GB: F8_E4M3 rows with F32 per-row amax scales, meant to be mmapped from NVMe.
- The vision tower and merger are in the pack (`vit.*`, plain tensors).
- Names are Tencent's checkpoint names (`learnable_sink_param`, `linear_gate`, `hc_attn_layer.hc_pre.hc_*`,
  `routed_expert_down_proj`, per-expert `experts.E.{gate,up,down}_proj`), not transformers' renames. 254,613 tensors:
  1,595 plain tensors, 63,253 EXL3 projections (`trellis`/`suh`/`svh`/`mul1`) and 6 plain projections (layer 0's
  attention and dense MLP). Engram tables: `...multi_head_embedding.embedding.weight` (F8_E4M3) + `.scale_bf16`.
  `pack.zig` checks all of it against the config (0 missing, 0 unknown, 0 partial on the real index).
- Gated (manual approval) on Hugging Face.

Memory: 93 GiB of weights + ~1 GiB ViT + KV/activations do not fit beside GLM-5.3 prod (~110 GB). Hy4 runs need
GLM stopped. The Engram sidecar lives in the page cache and competes with the weights in unified memory: random
row reads (16 heads x 4 layers x tokens, 96 B each) go through direct I/O from NVMe, not the page cache.

## What TensorFold already has

| need | where |
|---|---|
| REXL3/SAGE trellis loader and matmuls | `feat/glm53-rexl3` (rebased: `feat/glm53-rexl3-rebased`) |
| MLA with kv_lora 512 / nope 192 / rope 64 / v 256 | GLM family (`glm`, `glm_dsa`), same dims |
| a sparse token selector (indexer + top-k) | `glm_dsa` (different scoring: per-token keys, no pooling) |
| hyper-connections, Engram with NVMe direct I/O, EXL3 | upstream `deepseek_v41` (Sinkhorn HC; different Engram hash/gating) |
| latent MoE (pre/post expert projections) | upstream `nemotron` |
| SigLIP2 tower on CUDA, picture helper | upstream `cuda/vit.zig`, `lfm2/vision.zig`, vision helper |
| MTP/draft decoding | DFlash drafter path |

## Gaps (new work)

1. Config and loader: `hy_v4_flash` config (done, `zig/src/families/hy4/config.zig`), tensor name map
   (PR conversion mapping: `learnable_sink_param`->sinks, `linear_gate`->gate_proj, `post_fc2_norm`,
   `routed_expert_{down,up}_proj`->pre/post expert proj, `hc_*_layer.hc_pre.*`->attn_hc/ffn_hc, `vit`->vision tower),
   the pack's REXL3 tensor layout.
2. iHC: sigmoid pre/post mixing (no Sinkhorn) and hc_head: a variant of deepseek_v41's HC kernels.
3. Attention: sigmoid output gate, sinks in the softmax, post-projection norm; sliding window 512 on the MLA latent
   cache (ring of 512 latents for 30 layers), M-RoPE interleaved for the rope part.
4. Compressed indexer: group pooling kernel (4 tokens -> 1 key with value/gate projections + APE), top-511 groups,
   then sparse MLA over the selected 2048 tokens; reuse glm_dsa's sparse attention gather.
5. Engram: the hash (done, host side, matches the reference), FP8 row decode + per-head RMSNorm, value/key
   projections, gating, per-HC scales; sidecar reader on deepseek_v41's direct-I/O path.
6. Latent MoE with 512 experts / top-16 and the shared expert, REXL3 grouped matmul.
7. Vision: tower on `cuda/vit.zig` (exact GELU, LayerNorm bias, learned grid resample), the 2x2 conv merger and
   newline column; preprocessor (smart resize + bicubic + patchify: grid math done, bicubic to port).
8. Chat: the template uses `str.format` (now in the engine) and `namespace`; it renders byte-identical to jinja2
   (tools/hy4/template_parity.py + zig/src/core/template/hy4_parity_test.zig). `reasoning_effort` must be `high` or
   `no_think` (the template raises otherwise), so the server must map thinking on/off to those; `hy_v4` tool/reasoning
   parser still to write.
9. DFly2 drafter (later; serve without speculation first).

## Plan

| step | needs GPU | status |
|---|---|---|
| config parser, refusals, fixture | no | done (`test-hy4-host`) |
| Engram hash ids vs reference | no | done |
| picture grid: smart resize, token count, 3D positions, position-grid taps (vs torch), patch order | no | done |
| tensor-name map (pack -> transformers) + pack index check against the config: `pack.zig`, real index slice fixture + on-demand full check | no | done |
| chat template render vs jinja2 (text, pictures, tools, reasoning modes) | no | done |
| picture preprocessing (smart resize, core resample `.torch` = torchvision antialiased uint8 bicubic, fused normalize, patches) vs HYV4VLImageProcessor: `preprocess.zig` | no | done |
| CPU reference of one decoder layer vs transformers (iHC, gated MLA + sinks + gate, M-RoPE, sliding window, compressed indexer group recall, dense MLP): `layer_ref.zig`, fixture from tools/hy4/layer_fixture.py | no | done |
| reply markup onto the GLM tool/thinking parser, reasoning_effort mapping (thinking on/unsaid -> high, off -> no_think) | no | done (`reply.zig`); server wiring next |
| CPU reference of the latent MoE block (sigmoid router + bias, top-k, latent pre/post projections, clamp, shared expert) | no | done |
| CPU reference of the Engram block (document-bounded n-gram windows from the time plane, hash, FP8 E4M3 round trip, per-head norms with f32-eps RMSNorm, gated per-stream residual) and the decoder layer carrying it: `engram_ref.zig` | no | done |
| CUDA kernels: iHC, pooled indexer, sinks/gate, SWA latent ring, Engram gather | small tests only, beside prod | |
| ViT tower + merger vs transformers on CPU (layers extracted from the base repo) | short | |
| full model load and first tokens | yes, GLM stopped | needs a window |
| multimodal serve, conformance, DFly2 | yes | |
