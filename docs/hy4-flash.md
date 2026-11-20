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
| reply markup onto the GLM tool/thinking parser, reasoning_effort mapping (thinking on/unsaid -> high, off -> no_think) | no | done (`reply.zig`); server wiring done: `server/hy4_reply.zig` rewrites Hy4 tags to the GLM forms on decode and maps GLM tag lookups to Hy4 ids (`hf_text.zig`), `prompt.hy4Effort` |
| CPU reference of the latent MoE block (sigmoid router + bias, top-k, latent pre/post projections, clamp, shared expert) | no | done |
| CPU reference of the Engram block (document-bounded n-gram windows from the time plane, hash, FP8 E4M3 round trip, per-head norms with f32-eps RMSNorm, gated per-stream residual) and the decoder layer carrying it: `engram_ref.zig` | no | done |
| load plan (every pack tensor -> GPU arena or NVMe Engram sidecar, EXL3 shapes/widths checked; whole pack: 63,253 EXL3 groups, GPU 99,754,676,490 B, NVMe 64,680,628,768 B) and device layout (92.9 GiB arena, per-layer slots): `load_plan.zig`, `device_layout.zig` | no | done |
| CUDA loader: one arena, per-part uploads, EXL3 groups as `cuda.exl3.Weight` (mul1, stored strides): `cuda.zig` (`test-hy4-cuda-host`); whole-pack load + read-back spot checks: `cuda_load_test.zig` (HY4_LOAD_PACK) | the load: yes, GLM stopped | host tests done; load needs a window (`tools/hy4/window/window.sh`) |
| CUDA kernels: iHC pre/post (`kernels/cuda/hy4/hc.cu`, `cuda_hc_test.zig`) | small tests only | done, passed on the GB10 in the first window |
| forward kernels (`kernels/cuda/hy4/forward.cu`, wrappers `fwd.zig`): GEMM/GEMV, RMS/LayerNorm, SwiGLU (clamp), GELU, M-RoPE, absorbed-q MLA with sinks + sigmoid gate + sliding window or indexer keep map, pooled indexer (pool, score, top-k), sigmoid router + bias, expert gather/scatter/combine, Engram gate, ViT attention, merge gather, newline column, iHC head | small tests vs `layer_ref`/`engram_ref`/`vision_ref` (`cuda_fwd_test.zig`, HY4_FWD_FATBIN): toy layers (sliding + dense MLP, indexed + MoE) over 1-5 chunk plans, Engram, a 4x6 vision grid | written; GPU run in the next window (prod MemAvailable stays under 4.5 GiB) |
| GPU model (`gpu_model.zig`: chunked prefill + decode over f32 activations, EXL3 linears via `cuda.exl3.Dense`, per-layer latent/rope/index caches; `gpu_vision.zig`: tower + merger) and pack binding (`bind.zig`: arena slots -> layers, head, tower; e4m3 table) | no to build (`test-hy4-cuda-host` compiles them) | done |
| serve engine (`serve_engine.zig`: native_engines `hy_v4_flash/exl3`, one lane, Engram rows by pread from the NVMe sidecars, `--vision` with the template's start/pad/end markers, pad expanded to H x (W+1) tower rows) and server `tensorfold-hy4` (`zig build hy4-serve -Dfatbins=DIR -Dhy4-nvcc=NVCC [-Dtarget=aarch64-linux-gnu.2.39]`) | to run: yes, GLM stopped | built (x86_64 + aarch64), staged on the Spark (`~/hy4gpu/tensorfold-hy4`); first run in the next window |
| full model load and first tokens, text + image smoke | yes, GLM stopped | whole-pack load + spot checks passed (first window); `tools/hy4/window/window.sh` now runs the kernel tests, then `tensorfold-hy4 serve --vision` on 127.0.0.1:8095 and `smoke.py` (text, image, thinking) |
| multimodal serve, conformance, DFly2, speed (see the optimisation plan) | yes | |

## Optimisation plan (after the first served tokens)

The first engine (`serve_engine.zig`) is written for correctness: f32 activations, a full-length f32 latent cache in
every layer, the MoE grouped on the host (expert ids downloaded, one sync per MoE layer), Engram rows read by `pread`
per token and head, 256-row prefill chunks, one stream, no drafts. Each step below is measured in a window before and
after (prefill tok/s at 2k/16k/64k, decode tok/s on prose/code/French, MemAvailable low), and kept only if the kernel
tests and a fixed greedy transcript stay identical (or within the tested tolerance where precision changes).

Memory and context first, since it decides streams and context:

1. Caches: the 30 sliding layers keep only a 512-row ring (window 512); only the 12 sparse layers keep the full
   history. Latent + rope rows in bf16 (576 values), indexer pooled keys in bf16. Arithmetic: f32 everywhere today is
   ~96.8 KB/token (3.2 GB at 32k); ring + bf16 is ~13.8 KB/token of latents + ~0.8 KB/token of pooled keys, i.e.
   ~3.8 GB per 262,144-token stream, ~7.7 GB at 512k, ~15.3 GB at 1M. Beside the 92.9 GiB arena that is roughly
   4 x 262k, 2 x 512k or 1 x 1M on the Spark: to be measured (boot margin, MemAvailable low, vision tower, DFly2).
2. Several lanes: `--parallel N` with batched decode rows (the GEMV path already takes m <= 8 rows) and per-lane caches.

Decode:

3. MoE grouping on the device (count + prefix + scatter kernel) instead of downloading the router's ids: removes 41
   host syncs per token; then the expert projections as one grouped EXL3 launch per MoE layer (the GLM/DSV4.1 grouped
   kernels) instead of one `Dense.decode` per active expert.
4. Engram: the n-gram ids depend only on input tokens, so the four Engram layers' rows (16 heads each) are read in one
   batched direct-I/O submission at step start, overlapped with the layers before L01; page-cache pressure off
   (`O_DIRECT`, aligned 98-byte rows in 4 KiB reads).
5. CUDA graph of the decode step once no host sync remains; bf16 activations in the attention/MLP inner products,
   f32 residual and hyper-connection streams kept.
6. DFly2: the pack's 5-layer drafter (`model.dfly_layers.*`, already in the load plan as group `draft`): draft k
   tokens, verify them in one target pass through the chunked-prefill path (t = k + 1 rows), roll back by cache length
   (ring rows and indexer pools rewritten from the accepted position; Engram windows recomputed). Adaptive k from the
   acceptance rate. On GLM the DFlash2 gain was largest on code/math/JSON and smallest on prose; same measurement here.

Prefill:

7. Chunks of 2048 rows (work buffers sized for it); plain BF16 projections on tensor cores (mma bf16) instead of the
   scalar GEMM; EXL3 experts through the existing prompt kernels, FP8 as on GLM if it holds the tolerance.
8. Attention prefill tiled flash-style over the absorbed MLA (sliding: 512-row window; sparse: the indexer's top-2048
   keep map per row), indexer scoring as a tensor-core GEMM over pooled keys.
9. Vision: bf16 tower with tiled attention for large images; the tower runs once per image hash (prompt cache).

Order: 1 -> 3 -> 4 -> 7 (biggest expected wins: context, per-token syncs, Engram I/O, prefill GEMM), then 6 (DFly2),
5, 8, 9, 2. Streams at 256k / 512k / 1M are measured after 1 and 2.

### Window 2 (19 Nov 2026, first engine, 32k, one stream)

Text + image smoke 3/3. `bench.py`: short prompt 72 tok, TTFT 1.25 s, decode 9.73 tok/s; 2k prompt (1820 tok) TTFT
17.6 s (103 tok/s prefill), decode 8.74 tok/s; 8k prompt (7178 tok) TTFT 78 s (92 tok/s prefill), decode 8.15 tok/s.
Load 94,342 MiB in 170 s; 16.2 GiB of the device free after load with 3.2 GiB of f32 caches at 32k.

## Optimisation status (code; GPU-tested in the next window)

Done in code, host-tested, compiled for sm_121 / aarch64; none of the new kernels has run on a GPU yet:

1. **Caches** (`gpu_model.Shape`, `hy4_cache_store`, `hy4_mla_attn`): the 30 sliding layers keep a ring of
   `align16(512 + chunk)` rows; the 12 indexed layers keep the full history. Latent + rope rows in f32, bf16 or fp8
   e4m3 (one f32 scale per row, amax/448) — `HY4_CACHE_DT=f32|bf16|fp8`, default bf16. The indexer's token-key rows are
   a small ring (`chunk + kpool`), pooled keys (one per 4 tokens) in bf16; top-k returns a compact ascending list of
   kept groups and attention visits only those (plus the row's own group). Per sequence (host test `capacity.zig`):

   | context | f32 | bf16 | fp8 |
   |---|---|---|---|
   | 256k | 7.18 GiB | 3.59 GiB | 2.09 GiB |
   | 512k | 14.31 GiB | 7.16 GiB | 4.17 GiB |
   | 1M | 28.56 GiB | 14.29 GiB | 8.32 GiB |

   With ~16 GiB free after load and a 4 GiB reserve (`HY4_RESERVE_MIB`): 1M only in fp8 (1 stream), 512k fp8 ~2-3,
   256k fp8 ~5-6, 256k bf16 ~3. The serve logs `hy4 capacity:` lines and refuses a `--parallel` that does not fit.
2. **Several streams** (`serve_engine` lanes, `gpu_model.layer(segs)`): each step packs every lane's decode row, then
   prompt pieces round-robin up to the chunk; projections, MoE and Engram run over all rows at once, rope / cache store /
   indexer / attention per segment. Tested on the GPU path by a two-sequence test (lag 0 and 8) against single runs.
3. **MoE on the device** (`moe_grouped.zig`): the router's picks feed the existing EXL3 grouped-expert kernels
   (`exl3_experts`, fp32 SwiGLU with the model's clamp) for up to `HY4_MOE_GROUPED_ROWS` rows (default: the chunk), no
   host sync; the per-expert path stays as fallback (`HY4_MOE_GROUPED=0`) and `HY4_MOE_CHECK=N` compares both on the
   first N calls.
4. **Engram** (`engram_batch.zig`): the rows of all four Engram layers for a step are read by a thread pool
   (`HY4_ENGRAM_THREADS`, default 32) and uploaded once; serial `pread`s were ~2.5 s per 256-row chunk.
5. **Prefill GEMM**: plain bf16 / f16 projections of >= 32 rows run on tensor cores (`hy4_gemm_tc_*`, wmma 16x16x16,
   64x64 tiles, fp32 sums; x rounded to the weight type). `HY4_TC_GEMM_ROWS=0` turns it off. Default chunk 256
   (`HY4_CHUNK`).

Deferred:

- **CUDA graphs** of the decode step: needs positions and segment tables on the device (today `p0`, ring slots and the
  segment loop are host values baked into launch arguments) and fixed-size grids for top-k / grouped MoE. Plan:
  (1) a device `Seg` table `{p0, rows, lane cache base, ring slot}` per segment, written into the pinned stage and
  uploaded with the ids; (2) rope / cache store / indexer / attention read it (grid over segments x heads, `p0` from the
  table, key counts bounded by the window or `(recall + 1) * kpool`, so grids are fixed); (3) grouped MoE with a fixed
  grid over `rows * topk` (empty tiles exit) — already true for the decode rows; (4) capture one graph per lane count
  (1..8) on the first decode step, replay with only the stage upload before it. Worth it once the profile shows the
  decode step's launch gaps: ~42 layers x ~25 launches, at a few us each ~3-5 ms of the ~70-90 ms step.
- **Multi-stream CUDA streams** (overlap of Engram reads / uploads with the first layers): after graphs.

### DFly2 drafter (design; implementation next)

Reference: Tencent AngelSpec (`models/draft/dfly2.py`, `dspark.py`, `hy4_mla.py`, config
`dfly2_hy_v4_flash_draft_config.json`), Apache-2.0 — our implementation follows its published design and credits it.
The pack carries `model.dfly_layers.0..4.*` (EXL3 q_b / linear_gate / o_proj / MLP gate-up-down 3072 -> 9216 -> 3072,
plain bf16 q_a / kv_a, f16 kv_b `[32 x (192 + 256), 512]`, norms, f32 learnable sink `[32]`) plus `mtp_fusion`
(eh_proj `[3072, 6144]`, enorm, hnorm, gate), `hidden_correction` (gate / up `[3072, 6144]`, down `[3072, 3072]`,
hidden_norm, embed_norm), `confidence_head` (`[1, 3072]` + bias) and `final_norm`. `draft_plan.zig` lists every one
with the shape the engine binds and checks them against the pack's headers (host test over
`fixtures/dfly_headers.json`, the 177 real headers: all bound, none left over).

- Inputs: the target's hidden states of the last accepted row (fused by `mtp_fusion`: enorm(embedding) ‖
  hnorm(hidden) -> eh_proj, gated), `hidden_correction` on top.
- 5 gated-MLA layers; layer i attends over the target's KV latents of layer 25 / 29 / 33 / 37 / 41 and reuses that
  layer's DSA top-2048 picks (no own cache, no own indexer), with a learnable sink and a sigmoid output gate.
- Drafts a block (SGLang uses 6) with `mask_token_id` 120009 for the not-yet-known positions; `confidence_head`
  gives a per-position stop score (adaptive k).
- Verify: the block goes through the multi-row path as one segment (t = k + 1 rows); rollback = lane position reset
  (ring rows and indexer pools are rewritten from the accepted position; the full-history layers just overwrite).
- Per draft layer, the keys are the target layer's cached latents (normed latent + roped key, as our cache already
  stores them) kept to that layer's top-k, plus the block's own latents from the draft layer's `kv_a` (the block sees
  itself in full); the attention is the same absorbed MLA as `hy4_mla_attn`, with a second key source.
- Engine work: a `Draft` struct bound from the pack (bind.zig group `draft`, shapes per `draft_plan.zig`), a forward reusing `hy4_mla_attn` with the
  target's `CacheView` of the five source layers, and the accept loop in `serve_engine` (greedy first, then
  sampling with the standard speculative acceptance).

### Window 3 (20 Nov 2026, 07:50-08:35 Paris; build 210e7b76c, bf16 caches at 32k, fp8 for capacity)

- Kernel tests: hc, Engram, vision, grouped-MoE tables, Engram batch, capacity passed. The forward-layer and
  tensor-core GEMM tests failed at first, and not the same way twice. The cause was the test harness, not the kernels:
  `Gpu.alloc` zeroed buffers with a legacy-stream `cuMemsetD8`, which does not order with the test's non-blocking
  stream, so the memset sometimes landed after the uploads and kernels (whole outputs or caches zeroed). With the memset
  on the test's stream every f32 plan matches layer_ref to ~4e-6 (sliding + dense and indexed + MoE, whole, chunked,
  decode rows, two sequences with lag 0 / 8), and the TC GEMM is within 9.5e-4 (bf16) / 1.3e-4 (f16) of the fp32 GEMM.
  The bf16 caches are within 2e-2 and fp8 within 0.3 on the random toy layers, i.e. bf16 x 16 for the coarser
  mantissa (tolerances set to 3e-2 / 0.4). The serve's boot-time cache memset now also finishes before use.
- MoE: grouped device MoE vs per-expert, `HY4_MOE_CHECK=4`: worst 1.9e-4 .. 2.4e-4 of the largest value (34 rows).
- Smoke 3/3 at 32k bf16 (text 'Paris', image 'Red', thinking '17 * 3 = 51'), and 3/3 again on the 512k fp8 server.
- bench (32k, bf16, 4 lanes; window 2 first engine in brackets): short 72 tok TTFT 0.57 s (1.25), decode 13.9 tok/s
  (9.7); 2k TTFT 14.6 s, 125 tok/s prefill (103), decode 11.8 (8.7); 8k TTFT 64 s, 112 tok/s prefill (92), decode 11.5
  (8.2). 4 streams at once: 27.5 tok/s aggregate, 7.1-7.9 tok/s each.
- Long prompt (28,553 tokens, fp8): TTFT 266 s (107 tok/s prefill), decode 10.9-11.4 tok/s. Prefill is still the
  bottleneck: the batched Engram reads and TC GEMM gave ~+20%, not the hoped-for multiple. Next: profile a prefill
  chunk (attention over the full sparse history and the per-expert EXL3 path for 256-row chunks above
  `HY4_MOE_GROUPED_ROWS` are the suspects).
- Streams measured (boot with ~19 GiB free beside a 4 GiB reserve; engine log `hy4 capacity`):

  | context | f32 | bf16 | fp8 | booted (fp8) |
  |---|---|---|---|---|
  | 256k | 2 | 4 | 6-7 | 7 lanes, 2142 MiB each, MemAvailable low 6.9 GiB with 4 streams + a 28k fill |
  | 512k | 1 | 2 | 3 | 3 lanes, 4269 MiB each, MemAvailable 9.4 GiB after the fill |
  | 1M | 0 | 1 | 1 | 1 lane, 8523 MiB, 10.3 GiB of the device free |

### After window 3 (code; GPU-tested in the next window)

- **Tensor-core attention for prompt chunks** (`hy4_mla_scores_tc` / `hy4_mla_softmax` / `hy4_mla_pv_tc`): for >= 16
  rows (`HY4_ATTN_TC_ROWS`, 0 = the scalar kernel), scores Q_abs x [latent | rope]^T as wmma bf16 64x64 tiles over the
  row's keys (ring window, the indexer's kept groups plus its own, or causal), the softmax with the head's sink, then
  P x latent in wmma; scratch for `HY4_ATTN_TC_PIECE` rows (64) x heads x keys. f32 / bf16 / fp8 caches read through
  the same per-row scale. Tested against the scalar kernel (sliding ring bf16 / fp8, indexed list bf16 / f32, dense
  causal; 2e-2 of the largest output) in `hy4-gpu-test`.
- **Profiling mode** (`prof.zig`, `HY4_PROFILE=N`): CUDA events between op groups of every step (upload, embed,
  Engram, attention projections, rope + cache store, indexer, attention, attention output, router, experts, shared
  expert, dense MLP, logits) and host timers (schedule, Engram reads, sampling); every N steps the serve logs
  `hy4 profile prefill|decode: steps, rows a step, wall ms, device ms: <op> ms ...; host ...`.
- **Fewer syncs on the step**: the host inputs (embeddings, masks, ids, positions) live in two pinned stages used in
  turn, so the uploads are async and the stream no longer syncs after them; a stage is reused once the event after its
  uploads fired.
- **Larger chunks**: `HY4_CHUNK` up to 1024 (the indexer's logits are `[chunk, ctx / 4]` f32: 1 GiB at chunk 1024 and
  1M context, so 512 is the next step at 32k-256k).
- **Window script**: a profiling pass after the smoke + bench: one 32k boot per `HY4_PROF_VARIANTS` entry
  (`CHUNK:ATTN_TC_ROWS`, default `256:16 512:16 256:0`), a ~6k-token prompt and 4 streams, the profile lines in the
  window log.
