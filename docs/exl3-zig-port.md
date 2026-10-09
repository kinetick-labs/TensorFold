# Reading EXL3 packs in the native Flash Next CUDA backend

A port plan, not production code. Target: the Zig engine's Flash Next CUDA path
(`zig/**`, family `qwen4_exp`) reading an ExLlamaV3 **EXL3** pack
(`quant_method: "exl3"`, codebook `mul1`) — the pack the Spark rig serves as
`--exl3-405-turboderp` (`/home/jct-spark/models/exl3-405`, 9 shards + a 39 GB
n-gram table). All paths and line numbers are exact.

- **Reference sources (Python line):** branch `upstream/python-0.6`, head `ed78d6f`.
  Cited as `file:line` at that head (read via `git show upstream/python-0.6:<path>`).
- **Target tree (native line):** branch `feat/zig-exl3` off `integration/1.0.2`
  (head `8d955e4`). Cited as `zig/...:line` in the working tree.
- **The constraint:** no Python at serve time. Anything that cannot compile to a
  fatbin, or that drags torch/Triton in, is not reusable as-is.

The single most important finding is in §2: **the EXL3 arithmetic is pure CUDA
C++ and can be compiled into a fatbin exactly like the kernels the tree already
ships.** The work is (a) telling the engine how to read the format, (b)
re-authoring the host launchers in Zig, and (c) rescuing the residency gate,
which today counts a 36 GiB table the engine already mmaps but does not hold
(§4). The Triton prompt kernel is the one genuine rewrite.

---

## 1. What the pack demands

### 1.1 `quantization_config.json` — structure and granularity

Top level is a **dict of 9 keys**: a global header plus a per-**module** stub.

| key | type | value in this pack |
|---|---|---|
| `quant_method` | str | `"exl3"` |
| `version` | str | `"1.4.4"` |
| `bits` | float | `4.05` (the average over the model, not per-tensor) |
| `head_bits` | int | `6` |
| `calibration` | dict | `{"rows": 250, "cols": 2048}` |
| `out_scales` | str | `"always"` |
| `codebook` | str | `"mul1"` |
| `mtp_bits` | int | `4` |
| `tensor_storage` | dict | **74,395 entries** |

Granularity is **global header + per-module (prefix) dict** — `tensor_storage`
is keyed by the module path *without* a `.weight`/`.trellis` suffix
(`model.language_model.layers.0.mlp.experts.10.down_proj`). There are **exactly
two per-entry key-sets**:

- **74,041 quantized modules**: `{stored_tensors, quant_format, bits_per_weight,
  mul1_multiplier}` — `quant_format == "exl3"`, `bits_per_weight ∈ {4, 6}`
  (73,740 modules at 4, 301 at 6), `mul1_multiplier` constant `2212286765`.
- **354 unquantized modules**: `{stored_tensors}` only (`embed_tokens`,
  `hc_expand`, `*_hyper_connection`, `mlp.gate`, `mlp.shared_expert_gate`,
  `*_norm.weight`, `in_proj_a/b`, `A_log`, `dt_bias`, `ple.key_proj/value_proj`).

**The bits that matter are per-tensor and are NOT in the header.** The scalar
`bits`/`head_bits`/`mtp_bits` in the header are the model's *average* and the two
fixed heads; the actual per-layer width is read from the **trellis shape**:
`last_dim / 16`. This is exactly what `format.py` does — `bits_of` (`format.py:86`),
wired through `parse_group` (`format.py:226`). A port that trusts the scalar
`bits` will mis-read every 6-bit dense layer. The per-module `bits_per_weight`
in `tensor_storage` is a second, redundant source (and it correlates, e.g. all
301 six-bit modules are the shared_expert/dense-attn/GDN/lm_head set).

There is **no** `tensor_storage` entry carrying `version`, `head_bits`,
`codebook`, `mtp_bits`, or `quant_method` — those are top-level only.

`config.json` also carries the **same block nested** (`quantization_config`,
minus `tensor_storage`), which is what the engine actually reads at load time
(`format.py:config_fields`, `format.py:281`; the reader looks under
`quantization_config`/`quantization`, top level or `text_config`). The Python
gate `require_config` (`format.py:293`) refuses a codebook outside
`{3inst, mcg, mul1}`, a non-integer `head_bits`/`mtp_bits`, or an average `bits`
outside 1 to 8 — **before any weight is downloaded**. (It does not look at
`out_scales`: the header only carries it, `format.config_fields` returns it and
nothing validates it.)

### 1.2 Tensor-name suffixes and their meaning

`format.py:22` names the parts: `PARTS = ("trellis", "suh", "su", "svh", "sv",
"mcg", "mul1", "bias")`. This pack uses a subset:

| suffix | dtype / shape | meaning |
|---|---|---|
| `.trellis` | I16 `[K/16, N/16, 16*bits]` | the quantized weights: one 16×16 tile per `(K/16, N/16)`, each a 16-bit-state bitstream |
| `.suh` | F16 `[K]` | input-side scales `diag(suh)` in the rotated domain |
| `.svh` | F16 `[N]` | output-side scales `diag(svh)` |
| `.mul1` | I32 `[]` (scalar) | codebook **marker**, value `0x83DCD12D`; its presence selects the `mul1` codebook |
| `.weight` | F16/BF16 | plain (unquantized) tensors: `embed_tokens`, router gates, norms, `mtp.fc_*` … |
| `.A_log`, `.dt_bias` | F32 | GDN/DeltaNet constants |

Suffix census over the index's 303,121 tensors: **7** — `.trellis`, `.suh`,
`.svh`, `.mul1` (75,587 each), `.weight` (701), `.A_log` (36), `.dt_bias` (36).
**Absent from this pack:** `.su`/`.sv` (packed ±1 sign-words, the alternative to
full fp16 scales — `format.py:unpack_signs`, `:135`), `.bias`, `.mcg`, and the
`3inst`/`mcg` codebooks. The loader should still accept `su`/`sv`/`mcg`/`3inst`
for generality, but this pack needs only `trellis`/`suh`/`svh`/`mul1`.

The reconstruction the kernels compute is
`W = diag(suh) · H_K · W_q · H_N · diag(svh)` (`format.py:dequantize`, `:166`),
with `H` the 128-point Hadamard (`format.py:hadamard`, `:146`), and the forward
`y = rotate(x·suh) @ W_q` then `rotate(...)·svh` (`format.py:forward`, `:174`).

### 1.3 Typical dimensions (from shard headers, no data loaded)

Convention: `trellis = [K/16, N/16, 16*bits]`, `suh = [K]`, `svh = [N]` fp16,
`mul1` scalar. Model: `hidden 2560`, `num_experts 512`, `moe_intermediate 640`,
`vocab 248320`, 48 layers.

**MoE expert** (layer 0, expert 10):
- `gate_proj.trellis = [160, 40, 64]` → K=2560, N=640, **bits 4**; `suh=[2560]`, `svh=[640]`.
- `up_proj.trellis = [160, 40, 64]` → same.
- `down_proj.trellis = [40, 160, 64]` → K=640, N=2560, **bits 4**; `suh=[640]`, `svh=[2560]`.

**Dense linears** (6-bit set):
- `self_attn.q_proj.trellis = [160, 768, 96]` → K=2560, N=12288, **bits 6**.
- `self_attn.k_proj/v_proj.trellis = [160, 32, 96]` → N=512.
- `self_attn.o_proj.trellis = [384, 160, 96]` → K=6144, N=2560.
- `linear_attn.in_proj_qkv.trellis = [160, 640, 96]` → N=10240.
- `mlp.shared_expert.{gate,up}_proj = [160, 40, 96]`, `.down_proj = [40, 160, 96]`.
- `lm_head.trellis = [160, 15520, 96]` → K=2560, N=248320, **bits 6** (`head_bits=6`).

**MTP** (`mtp_bits=4`): `mtp.fc_hidden/fc_embedding.trellis = [160, 160, 80]`
→ **bits 5**; `mtp.layers.0.self_attn.*` → bits 6; MTP experts → bits 4.

So the pack **mixes widths 4, 5 and 6**; a per-tensor decode is mandatory.

### 1.4 The n-gram table

Single consolidated file `ngram_embedding.safetensors`, header 920 B, tensors
under `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.`:

| tensor | dtype | shape |
|---|---|---|
| `.trellis` | **I16** | **`[320001536, 61]`** |
| `.head_bias` | F16 | `[16, 160]` |
| `.head_offsets` | I64 | `[16]` |
| `.head_vocab_sizes` | I64 | `[16]` |
| `.layer_multipliers` | I64 | `[3]` |

With `dh = 160`: `words_per_row = 61`, `bits = (61-1)*16 // 160 = 6`,
`1 + 160*6//16 = 61` ✓, row = 122 bytes, data = **39,040,187,392 B (36.36 GiB)**.
`__metadata__` names it `exl3_ngram_trellis`, `codebook=mul1`, `row_dim=160`.
It is **one `.trellis` tensor, not sharded** — so the naming (`...ngram_embedding.trellis`)
differs from the ModelOpt path's `...ngram_embedding.shard_i.weight` (see §3.3).

---

## 2. What already exists on `upstream/python-0.6`, reusable as-is

The decisive question is torch-dependence. Verdict per file:

| file | device code | TU as a whole |
|---|---|---|
| `cuda/exl3/decode.cuh` (162 ln) | **PURE CUDA C++** | **PURE** — includes only `<cstdint>`, `<cuda_fp16.h>` |
| `cuda/exl3/linear.cu` (349 ln) | **PURE** (`cuda_fp16`/`cuda_bf16`/`asm`) | MIXED — 3 `__global__` + all `__device__` are pure; the `exl3_*_cuda` host wrappers take `at::Tensor` |
| `cuda/exl3/experts_grouped.cuh` (356 ln) | **PURE** | MIXED — includes `<ATen/ATen.h>` but device code never uses it; `grouped_launch`/`dequant_launch` call `TORCH_CHECK` (`:339`, `:352`) |
| `cuda/exl3/experts.cu` (371 ln) | **PURE** | MIXED — same shape as `linear.cu` |
| `cuda/exl3/experts_cb{0,1,2}.cu` (7 ln each) | — | **PURE** — explicit template instantiations, no torch symbol |
| `cuda/exl3/linear.cpp` (74 ln) | — | **TORCH** — pybind11 host glue only |
| `cuda/exl3/experts.cpp` (141 ln) | — | **TORCH** — pybind11 host glue only |
| `cuda/exl3/prefill.py` (95 ln) | Triton JIT | **TRITON + torch** |
| `cuda/exl3/format.py` (386 ln) | numpy | pure Python (reference decoder) |
| `cuda/exl3/inspect.py` (99 ln), `__init__.py` | — | pure Python |

**The arithmetic kernels are torch-free CUDA C++ and drop into a fatbin
unchanged.** What is torch-bound is (i) the thin host launchers (argument
marshalling, grid/block/smem, `C10_CUDA_KERNEL_LAUNCH_CHECK`) inside the `.cu`
files, and (ii) the `.cpp` pybind glue. Those must be re-authored in Zig — but
they carry no arithmetic, only the launch contract, which is fully tabulated
below.

### 2.1 `decode.cuh` — the shared tile decoder (PURE, vendor verbatim)

Namespace `tf_exl3`; `Codebook {CB_3INST=0, CB_MCG=1, CB_MUL1=2}` (`:10`).

| symbol | line | role |
|---|---|---|
| `tile_words<K2>()` | `:13` | `4*K2` words a tile |
| `stream_end<K2>(p)` | `:19` | E(p), end bit of value p's window |
| `lane_words<K2>()` | `:25` | words a lane reads (2, or 3 for 3.5/5–8 bits) |
| `lane_start / load_lane_words / ldg_lane_words` | `:37 / :45 / :54` | lane's words of a tile |
| `window16 / lane_states` | `:62 / :71` | the 8 states a lane decodes |
| `decode2<CB>` | `:84` | 2 states → half2; MUL1 via `__dp4a`+`__hfma2` (consts `0x83DCD12D`, `0x1EEE`, `0xC931`), MCG/3INST via `__byte_perm`+`__hadd2` |
| `decode_lane<K2,CB>` | `:120` | the tile's two mma.m16n8k16 B fragments |
| `mma16816` | `:136` | inline PTX `mma.sync…m16n8k16` |
| `fwht128` | `:144` | in-register 128-point Walsh–Hadamard (`__shfl_xor_sync`) |
| `HAD_SCALE` | `:160` | `1/√128` |

No Hadamard matrix buffer is passed to the CUDA kernels — rotation is `fwht128`
in registers. (Only the Triton `_gemm` loads an explicit `H`.)

### 2.2 `linear.cu` (PURE device code + torch launchers)

Kernels (all **PURE**):

- `rot_in_kernel` `:112` — `__global__ __launch_bounds__(128)`, args
  `(const void* x, int x_dtype, const half* suh, half* xh, int K)`. Computes
  `xh = fp16(((x·suh)@H)/√128)`. Launch (`exl3_rot_in_cuda`, `:294`):
  `grid=((K/128+3)/4, M)`, `block=128`.
- `linear_kernel<K2,CB,WK>` `:128` — `__launch_bounds__(WK*32)`, args
  `(const half* xh, const uint32_t* T, long long stride_k, long long stride_nb,
  const half* svh, const half* bias, void* y, int y_dtype, float* Z,
  int* counters, int M, int K, int N, int SK)`. Buffers: `xh` fp16 `[M,K]`
  (K%128=0); `T` = the trellis viewed as uint32 words, `numel == K*N*K2/64`
  (`linear.cpp:47`), 16-byte aligned; tile `(kt,nt)` at
  `kt*stride_k + (nt/8)*stride_nb`; `svh` fp16 `[N]`; optional `bias`; `y` f16/bf16/f32
  `[M,N]`; optional `Z` fp32 `[SK,M,N]` for split-K; `counters` int32 left zero
  (cross-block reduction). Launch (`exl3_linear_cuda`, `:303`):
  `grid=(N/128, SK)`, `block=WK*32`, dynamic smem `= WK*min(M,8)*128*4`, lifted
  above 48 KB via `cudaFuncSetAttribute`. Split plan `plan(k,n)` (`linear.py:30`):
  `(SK,WK)` starts `(1,8)` when `N/128>=64`, else `(1,4)`, growing K-split for
  narrow layers.
- `unpack_kernel<K2,CB>` `:269` — `__launch_bounds__(32)`, `(T, W, N, stride_k,
  stride_nb)`, `W` fp16 `[K,N]`; one warp per tile; launch `grid=(N/16,K/16)`,
  `block=32`. This is the **dequantize-to-dense** path (used for the prompt and
  for any non-gemm consumer).

Template instantiation set to reproduce: `(K2,CB) ∈ TF_EXL3_ALL`
(`linear.cu:291` = all `CB{0,1,2}` × `{2,4,6,8,10,12,14,16}` plus the `X(3,2) X(5,2)
X(7,2)` half-width mul1 cases) × `WK ∈ {2,4,8}`. Codebook is a runtime `int`
selecting a compile-time `CB`.

`linear.py:Exl3Linear` (`:56`) is the family entry point: it repacks the int16
trellis into the word layout (`strips`, `:48`, or `stored`), unpacks `su`/`sv`
sign-words in Python (`format.unpack_signs`), and launches `rot_in`→`linear`.
**That repacking and the pointer/scales tables are Zig work (§3.3).**

### 2.3 `experts_grouped.cuh` + `experts.cu` (PURE device + torch launchers)

- `grouped_kernel<CB,NT,W,PF,LO,HI>` `experts_grouped.cuh:190` — the MoE GEMM.
  Buffers: `X0/X1` fp16 rotated inputs; `TP0/TP1` int64 `[E]` per-expert trellis
  **device pointers**; `K2_0/K2_1` int32 `[E]` half-bits per expert; `uids`,
  `ucount`, `members` `[nexp_max,maxm]` `(row<<5)|slot`; `Z` fp32 split partials.
  Device helpers `cb_pair` `:13`, `Fmt` `:42`, `LaneMap` `:54`, `fetch` `:70`,
  `decode_tile` `:82`, `warp_tiles` `:139`; `mma16816` `:115`. Launch configs
  `(NT,W,PF) ∈ {(8,4,1),(8,4,2),(4,4,2)}`, K2 ranges `{8,8}`/`{2..10}`/`{2..16}`.
- `dequant_kernel<CB,K2>` `:287` — one warp per tile; `grid=(K/16,N/16)`,
  block 32.
- `group_kernel` `experts.cu:19` — builds the dispatch tables (`pick`→`uids`/
  `ucount`/`members`); `rot_in_kernel<TIN>` `:93`; the epilogues
  `gateup_epilogue_kernel` `:116`, `down_epilogue_kernel` `:161`,
  `combine_kernel` `:183`, `down_combine_kernel` `:194`. All **PURE**; the `*_cuda`
  wrappers (`:252`–`:371`) are torch.

`experts.py:Exl3RoutedExperts` (`:55`) builds the per-expert pointer/K2 tables
and stacked scales (`prepare`, `:83`) and drives the launch sequence
(`routed`, `:173`): `group → rot_in → grouped(gate/up) → gateup_epilogue →
grouped(down) → down_epilogue|down_combine`. **Re-implement in Zig.**

### 2.4 What is Triton and must be rewritten (no CUDA C++ exists)

- `prefill.py:16` `_gemm(...)` — the whole prompt GEMM in the rotated domain:
  `OUT = ((xh@W_q)@H)·scale·svh + bias`, with an explicit `H` matrix; launch
  `grid=(cdiv(m,bm)*(n//BN),)`, `BM=128 BK=32 GROUP=8`, `num_warps=8 num_stages=4`
  (`:92`). `Workspace` `:53`, `matmul` `:79`. **~50 lines of Triton → a new CUDA
  kernel.**
- `families/qwen4_exp/cuda/exl3_mm.py` — Triton `_f16_mm` `:65`, `_reduce` `:91`,
  `_embed` `:261`, `_ple_rows` `:278`. The bf16 fallback GEMM, the split-K
  reduce, embedding-row gather and PLE rows. **Rewrite as CUDA/Zig.**

### 2.5 The WC / mid-M kernel does not exist at the reference head

`linear_wc.cuh` is **not** in `upstream/python-0.6` at `ed78d6f`, nor any
`WC`/mid-M identifier. The 1–128-row path at this head *is* the plain
`linear_kernel` + `plan()` above. The wide-column kernel
(`linear_wc.cuh`, `WcLayout`, `linear_wc_kernel`, a `cp.async`/`ldmatrix`
NS-stage ring) lives only on the rig's own branch `origin/perf/exl3-midm-wc`
(commit `73ee7f3`), which also grows `linear.cu` (+590), `linear.cpp` (+59),
`linear.py` (+8), `prefill.py` (+103). **Port the head set first; treat WC as a
separate, later optimization** — it targets the 17–128-row verify window, a
perf lever, not a correctness prerequisite.

### 2.6 Cost, honestly

- **Vendored unchanged into a fatbin:** `decode.cuh` (162 ln) + the `__global__`
  / `__device__` bodies of `linear.cu` (~250 ln), `experts_grouped.cuh` (~360 ln),
  `experts.cu` (~200 ln) ≈ **~970 lines of C++ moved, not rewritten.** The only
  edit is stripping the `ATen`/`torch` includes and the host launchers from the
  `.cu` (keep the device code, drop `TORCH_CHECK`/`at::` bits).
- **Re-authored in Zig** (no arithmetic, but real work): host launchers
  (grid/block/smem per §2.2–2.3), the trellis int16→int32 strips repack
  (`linear.py:48`), the per-expert pointer/K2/scale tables (`experts.py:83`),
  and the weight loaders (§3.3) ≈ **~800–1200 new lines**.
- **Genuinely new CUDA:** the `prefill.py` prompt GEMM and the `exl3_mm.py`
  Triton kernels ≈ **~400–700 new lines**, and the only place a real numerical
  kernel must be invented.
- **Not in the Python line at all:** the residency fix (§4), which is the make-or-break.

---

## 3. Where it hooks into the Zig engine

### 3.1 Kernel registry — `zig/build/cuda.zig`, `zig/src/cuda/kernels.zig`

The registry entry type carries **who** the source is, **how** it is flagged, and
whether the cubin is arch-tagged:

```zig
// zig/build/cuda.zig:6
const Kernel = struct { name: []const u8, flags: []const []const u8,
                        src: ?[]const u8 = null, arch_specific: bool = false };
```

- The array is `const kernels = [_]Kernel{ … }` at `zig/build/cuda.zig:11-50`;
  each entry names its source (`fn_int4`, `fn_pack`, …) and flags (`-O3`, …).
- `fatbin()` at `zig/build/cuda.zig:197-211` runs `nvcc -fatbin` with
  `torch_flags` (`:53`, `-std=c++20`, the half/bf16 operator guards,
  `--expt-relaxed-constexpr`), the per-kernel `k.flags`, `-gencode=arch=compute_{sm}{a},code=sm_{sm}{a}`
  (`sms` defaults `"121"`, `:106`), and reads the source from
  `zig/kernels/cuda/{k.src orelse k.name}.cu` (`:209`). Each becomes an
  anonymous import `fatbin_<name>` (`:80`).
- At runtime the blob is bound in `zig/src/cuda/kernels.zig`, e.g.
  `pub const fn_int4: []const u8 = if (available) &Blob("fatbin_fn_int4").bytes else &.{};`
  (`:45`), then loaded as `cuda.Module.load(d, cuda.kernels.<name>)` (cf.
  `cuda_weights.zig:176` for `fn_pack`).

**To add EXL3:** (1) an entry `.{ .name = "fn_exl3", .flags = &.{"-O3"} }` in the
array; (2) the source `zig/kernels/cuda/fn_exl3.cu` (the vendored device code
from §2); (3) a blob const `fn_exl3` in `zig/src/cuda/kernels.zig`; (4) if a
family launch wrapper is wanted, a `Kernels` binding in
`zig/src/families/flashnext/cuda_kernels.zig`. This is exactly how every existing
`fn_*` kernel is wired — **no new mechanism is required.**

### 3.2 Family admission and backend choice

- `zig/src/families/flashnext/cuda_native.zig`: `model_type = "qwen4_exp"` (`:21`)
  and `formats = &.{ "modelopt-nvfp4", "modelopt-mixed-precision", "gptq-b4" }`
  (`:26`) — the format strings the family admits. An EXL3 pack adds a string
  here (e.g. `"exl3-mul1"`).
- The `Family` type is `struct { model_type, formats }`
  (`zig/src/core/engine_api.zig:142`); the CUDA registry is
  `zig/src/native/cuda.zig:13` (`registry = .{ nemotron.native, flashnext.native }`),
  aggregated at `:16-21`, and **backend choice is by `model_type` string only**
  (`open()`, `zig/src/native/cuda.zig:322-328`). Since the EXL3 pack is also
  `model_type: "qwen4_exp"`, **the dispatch key does not change** — `formats` is
  descriptive/admission metadata, and the actual quant is read from the
  checkpoint's config and branched inside the family (`cuda_config.zig`, §3.4).

### 3.3 The CUDA weight loader — `zig/src/families/flashnext/cuda_weights.zig`

`pub fn load(...)` at `:1467` opens the checkpoint
(`core.Checkpoint.openModel`, `core/checkpoint.zig:62`), resolves shard paths,
and writes device buffers through the `Out` sink (`:298-360`): `begin(name,dtype,
shape)` allocates a `cuda.DeviceBuffer`, records a `Named`, and accumulates
`o.w.bytes += len`; `put` uploads/hashes; `end` checks length and finalizes the
digest. Every buffer carries the Python dataclass dotted path as its name, so
digests compare with the oracle's `weights.json`.

Name scheme: an opaque `prefix` (`language_model.`) + `mbase`
(`model.language_model.`/`model.`); tensors are fetched by name via `Loader.get`
(`:514`), overlaying extras. Quant formats the loader branches on, each a model
for an EXL3 sibling:

| format | function | line |
|---|---|---|
| bf16 linear | `linear` | `:540` |
| NVFP4 routed experts | `packRouted` | `:1059` |
| GPTQ int4 routed experts | `int4Experts` | `:990` |
| GPTQ int4 lm_head | `int4Head` | `:452` |
| FP8_PB_WO MTP experts | `mtpExperts` | `:1102` |
| block-FP8 dense | `fp8Face` | `:823` |

**Foreign-tensor policy — "read on purpose":** `skip()` (`:1441-1449`) walks
every tensor name under a `prefix` and inserts it into `ck.used`, so the final
`if (ck.unused() != 0) return error.UnusedCheckpointTensors;` (`:1624`) passes.
It is called for `"mtp."` (`:1620`) and `"model.visual."` (`:1623`). **EXL3's
`.trellis/.suh/.svh/.mul1` tensors are the weights, so they must be consumed by
real `get()`s that count as used — not skipped.** Only genuinely foreign sidecars
(e.g. a vision tower the engine does not read) go through `skip`. The "read on
purpose" mechanism is what keeps the `UnusedCheckpointTensors` gate from
rejecting a checkpoint whose top-level `*.safetensors` the loader deliberately
ignores (the gate sums *files*; the loader iterates *names*).

**EXL3 loaders to add**, mirroring the siblings: dense `X3` linear (compare
`linear` `:540`), routed experts (compare `packRouted` `:1059`/`int4Experts`
`:990`), `lm_head` (compare `int4Head` `:452`), MTP experts (compare
`mtpExperts` `:1102`), and the PLE table (`openTable`, §4). The layouts they
produce (the int16→int32 word strips, the per-expert pointer/K2 tables) are the
zig-side re-implementation of `linear.py:48` and `experts.py:83`.

### 3.4 The config reader — `zig/src/families/flashnext/cuda_config.zig`

`Quant` is `enum { mlx, modelopt, gptq }` (`:17`). `check()` (`:361-405`)
enforces `model_type ∈ {qwen4_exp, qwen3_8_flash_next}` (`model_types`, `:10`)
and refuses anything that is not ModelOpt or GPTQ:

```zig
// zig/src/families/flashnext/cuda_config.zig:369
if (self.quant != .modelopt) {
    why.set("the CUDA Flash Next engine reads the ModelOpt NVFP4 export; config.json's quantization is {t}", .{self.quant});
    return error.UnsupportedQuantization;
}
```

The `quant_method` dispatch is in `parse`:

```zig
// zig/src/families/flashnext/cuda_config.zig:800-813
const q = object(raw, "quantization") orelse object(raw, "quantization_config");
if (q) |qq| {
    const method = try str(a, qq, "quant_method");
    if (method.len == 0 or std.ascii.eqlIgnoreCase(method, "mlx")) c.quant = .mlx
    else if (std.ascii.eqlIgnoreCase(method, "modelopt")) { c.quant = .modelopt; c.modelopt = try modelOpt(a, qq); }
    else if (std.ascii.eqlIgnoreCase(method, "gptq")) { c.quant = .gptq; c.gptq = try gptqOf(a, qq, why); }
    else { why.set("quant_method {s}; Flash Next reads MLX or ModelOpt checkpoints", .{method}); return error.UnsupportedQuantization; }
```

**EXL3 inserts at four points:** (i) `Quant` enum `:17` — add `exl3`;
(ii) the `quant_method` dispatch `:800-813` — add the `"exl3"` branch, reading
`version`/`head_bits`/`mtp_bits`/`codebook`/`out_scales` (the header this pack
carries); (iii) a dedicated parser beside `modelOpt` (`:553`) / `gptqOf`
(`:586`); (iv) a `checkExl3` arm beside `checkGptq` (`checkGptqImpl` at `:412`),
which must **validate per-layer widths the way `format.py:require_config` does**
(codebook ∈ {3inst,mcg,mul1}; `head_bits`/`mtp_bits` integer ∈ BITS), and read
`self.bits`/`self.nvfp4_group` equivalents as it already does at `:814-818`.
Note the per-tensor width is not a config field — it is the trellis shape read at
load, exactly as `bits_of` (`format.py:86`) — so the config check is a coarse
gate and the loader does the precise per-tensor read.

---

## 4. The residency limit — the make-or-break

### 4.1 The gate

`zig/src/native/cuda.zig` sums the checkpoint **before loading** and refuses a
pack that does not fit:

```zig
// zig/src/native/cuda.zig:87-98
fn weightBytes(io: std.Io, dir: []const u8) u64 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return 0;
    … while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const st = d.statFile(io, e.name, .{}) catch continue;
        total += st.size;
    }
    return total;
}
```

It is **non-recursive**: every *top-level* `*.safetensors` — model shards, the
n-gram table, the vision sidecars — is summed. The budget is
`free − reserve` capped by an optional limit: `reserve =
budget.reserveBytes(TENSORFOLD_MEMORY_RESERVE_GIB, total)` (`:120`, default
`max(4 GiB, total/10)`, `cuda_memory.zig:49`), `limit =
budget.limitBytes(TENSORFOLD_CUDA_MEMORY_LIMIT_GB)` (`:124`), ceiling =
`Pool.room` (`cuda_memory.zig:42`). The refusal:

```zig
// zig/src/native/cuda.zig:388-390
if (weights > before.room(held0)) {
    problem.* = "the checkpoint's {d:.1} GiB of weights do not fit the {d:.1} GiB the CUDA memory budget grants (… free less a … GiB reserve{s}) …";
    return null;
}
```

A second gate runs after load: `budget.admit(room, loaded.stream_bytes, o.lanes, …)`
(`:414`).

**This pack against the gate** (top-level `*.safetensors`):

| file set | bytes | GiB |
|---|---|---|
| 9 model shards | 67,709,213,431 | 63.06 |
| `ngram_embedding.safetensors` | 39,040,193,720 | 36.36 |
| `vision-f16` + `vision_k6` | 1,459,288,189 | 1.36 |
| **sum `weightBytes` sees** | **108,208,695,340** | **100.78** |

That is the 100.8 GiB the release notes already flag: knife-edge against a
~104 GiB budget (114 free less a 10 GiB reserve) on a 128 GiB Spark that is not
empty. **But the table is not actually resident** (see §4.2). Excluding it, the
gate would see **69.17 GB = 64.42 GiB** — comfortable. The gap between 100.8 and
64.4 GiB *is* the residency problem.

### 4.2 How the n-gram table is treated today — the paging path already exists

The task's premise ("`cuda_ngram.zig` handles `ngram_embedding.safetensors`") is
half-right and worth stating precisely, because it changes the plan:

- **`zig/src/families/flashnext/cuda_ngram.zig` does not touch the file's bytes.**
  It is pure host id arithmetic: `splitmix64` (`:12`), the head sizes/offsets/
  multipliers, `NGram.init` (`:64`), `ids` (`:98`), `advance` (`:135`). It computes
  *which row* to read; it never reads a row.
- **The table's bytes live in `cuda_weights.zig`.** The `NgramTable` type
  (`cuda_weights.zig:117-165`) holds `files: ArrayList(st.File)` and
  `shards: ArrayList([]const u8)` — **mmap-backed slices**, no device copy:

  ```zig
  // cuda_weights.zig:142-148
  fn row(t: *const NgramTable, id: u64) []const u8 { … return t.shards.items[s][…]; }
  ```

  `openTable` (`:1247-1322`) calls `st.File.open(...)` (`:1282`) — and
  `core/safetensors.File.open` maps with `populate = false`
  (`zig/src/core/safetensors.zig:150`), i.e. **demand-paged, exactly Python's
  `np.memmap(..., mode="r")`**. `t.shards.append(tensor.bytes)` (`:1294`) stores
  slices into that map. `gather` (`:152`) reads one row a lookup at a time. This
  **is** the native analogue of the Python line's `cuda/ngram_pages.py` +
  `host_table.py` paging — clean-room, no Python.
- **On the GPU (world > 1)** `tableToGpu` (`:1325-1347`) copies the rank's heads'
  rows into a device buffer and a `[256]` LUT; those bytes **do** accrue into
  `Weights.bytes` via the `Out` sink (`:320`). At `world == 1` (the rig's case)
  **nothing is copied.**
- **The table's `.f8_e4m3`/`.bf16` restriction is enforced**: `openTable` refuses
  anything else with `error.UnsupportedQuantization` (`:1287-1290`). The EXL3
  table is **I16 `[320001536, 61]`** — **refused today.**

So the Python paging mechanism (`ngram_pages.py:Pins` `:68`, `lock_bytes` `:52`,
`Pins.runs` 1 GiB runs, `host_table.HostTable._random_access` `MADV_RANDOM`
`:394`, `_prefetch` `:408` over 8 threads) has a **partial Zig counterpart**:
the map + per-row gather exist; **the page locking/advice (`mlock`/`madvise`/prefetch)
does not**, and there is no budget-bounded `lock_runs`. That is a smaller port
than the release-note framing implies, but it is not zero.

### 4.3 What the EXL3 table needs

1. **Accept the consolidated trellis.** `openTable` (`:1254`) loops
   `L.c.ngram_shards` names of the form `{base}ngram_embedding.shard_i.weight`
   (`:1256-1257`). The EXL3 pack is one tensor `{base}trellis` (no shard index).
   Add the consolidated name and a single-shard path.
2. **Decode trellis rows.** `kind` currently accepts `.f8_e4m3`/`.bf16`
   (`:1287`); add an `I16` branch. A row is 61 int16 = 122 B encoding 160 values
   at 6 bits (`bits=(words-1)*16//dh`, `dh=160`, mul1) — the row decode mirrors
   `exl3_pack.NgramTable` (`qwen4_exp/cuda/exl3_pack.py:107`, `bits` at `:141`,
   `gather` at `:159`). `NgramTable.row`/`gather` (`:142`/`:152`) then produce
   bf16 `[160]` per row instead of an LUT/bf16 read.
3. **Rescue the residency gate.** This is the load-bearing change: **stop
   counting the mmapped table toward `weightBytes`** — as a per-file "mapped, not
   resident" allowance, or by summing only the files the loader will actually
   make resident (the shards + the sidecars that become device buffers). Done
   right, the pack's gate figure drops from 100.8 to 64.4 GiB and admission
   passes with room. **Keeping the table off-device at one rank is already the
   loader's behaviour (§4.2) — the gate simply does not know it.**
4. **(Optional) page locking.** If the rig wants the table's hot rows pinned,
   port the *shape* of `ngram_pages.Pins.runs` (`:94`, 1 GiB `RUN_BYTES`, charges
   only newly-missing pages) into a Zig `lock_runs(budget)` on `NgramTable`,
   using `std.posix.mlock` over page-aligned spans. Not required for correctness;
   required only for the throughput the Python line gets from pinned pages.

---

## 5. The plan, step by step — ordered, each with a proof

Order by dependency: **read the format first** (nothing else can be validated
without the per-tensor shape/bit logic), then the kernels, then the loader, then
residency, then the prompt kernel, then service. Each step names its *proof* and
whether the proof needs a GPU.

**Step 1 — Config and format reading.** Teach `cuda_config.zig` the `exl3`
`quant_method` and a `checkExl3`, reading `version`/`head_bits`/`mtp_bits`/
`codebook`/`out_scales`; and add a trellis-shape reader (`bits = last_dim/16`,
codebook from the `.mul1`/`.mcg` marker) — a Zig port of `format.py:86`/`:226`.
- *Proof (no GPU):* a Zig unit test on an `exl3` fixture (à la
  `fixtures_cuda_config*.json`) asserting `quant == .exl3`, codebook, and the
  derived bits for the 4/5/6-bit cases; and `tensorfold info <dir>` printing the
  family + quantization from the config alone.
- *Files:* `cuda_config.zig:17,361,798`; new fixture; `cli/info.zig`.

**Step 2 — The pure-CUDA EXL3 kernels into a fatbin.** Vendor `decode.cuh` and
the device bodies of `linear.cu`/`experts_grouped.cuh`/`experts.cu` into
`zig/kernels/cuda/fn_exl3.cu` (name `fn_exl3`), stripping the ATen/torch host
parts; register it (`build/cuda.zig:11`, `kernels.zig:45`).
- *Proof:* `zig build fatbins` produces an `fn_exl3` cubin for `sm121`;
  `cuobjdump` lists the expected instantiations; a GPU test that runs
  `unpack_kernel` on a synthetic trellis against a hand-computed reference.
- *Files:* `zig/kernels/cuda/fn_exl3.cu`, `zig/build/cuda.zig`, `zig/src/cuda/kernels.zig`.

**Step 3 — Host launchers in Zig.** Port `exl3_rot_in_cuda`/`exl3_linear_cuda`/
`exl3_unpack_cuda` and the `exl3x_*` launchers (grid/block/smem/`cudaFuncSetAttribute`
per §2.2–2.3) into the family's launch layer, plus `plan(k,n)` (`linear.py:30`).
- *Proof (no GPU):* a table test of `plan()`/grid arithmetic against
  `linear.py`'s outputs across the pack's (K,N) pairs.

**Step 4 — EXL3 weight loaders.** Add the dense `X3`, routed-experts, `lm_head`,
and MTP loaders to `cuda_weights.zig` (compare `int4Head:452`, `packRouted:1059`,
`mtpExperts:1102`), including the int16→int32 word strips repack and the
per-expert pointer/K2/scale tables.
- *Proof:* a `mode = .hash` load reproducing a Python-computed `weights.json`
  digest per buffer, plus `mode = .count` (no GPU); then `mode = .device` on the GPU.

**Step 5 — The n-gram trellis + paging.** Extend `openTable` to the consolidated
`.trellis`, add the 6-bit mul1 row decode to `NgramTable.row/gather`, and keep the
host map (no device copy at `world==1`). Optionally port `Pins.runs`.
- *Proof (no GPU):* a host test decoding a synthetic 61-word row to bf16 `[160]`
  and comparing against `format.py`'s reference; a header-load test reading one
  real row id from the pack and comparing the decoded row to the Python engine's.
  No device is touched.

**Step 6 — Fix the residency gate.** Make `weightBytes` (`native/cuda.zig:87`) not
count the mmapped table (per-file allowance / resident-only sum).
- *Proof (no GPU):* a unit test that the gate's byte figure for a fixture tree
  excludes the mapped table; a real admission test on the Spark with
  `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` set, reading the refusal/admission line.

**Step 7 — The prompt kernel (Triton → CUDA).** Write `fn_exl3_prefill.cu` for
`prefill.py:16`'s `_gemm` (`OUT = ((xh@W_q)@H)·scale·svh + bias`).
- *Proof:* a GPU test comparing the bf16 output to a host reference on small M.

**Step 8 — Service.** `tensorfold serve` the pack; one completion.
- *Proof:* the health route + one graded completion; the engine's own counters.

### What is NOT verifiable without the GPU

- Any kernel correctness: `decode2`/`decode_lane`/`mma16816`/`fwht128`,
  `linear_kernel`, `grouped_kernel`, the prefill GEMM. The Spark (or any `sm121`
  device) is required; the build itself (`nvcc`, `-gencode`) needs the CUDA
  toolkit.
- The residency admission **against real free memory** on the 128 GiB box —
  the gate arithmetic is testable, the admission is not.
- The end-to-end serve and any throughput number.
- The GPU n-gram path (`world > 1`, `tableToGpu` + `fn_pack.cu`): the rig runs one
  rank, so it is untested there.

### What IS verifiable without the GPU

- Config/format parsing and the per-tensor bits derivation.
- `plan()`/grid arithmetic against `linear.py`.
- The strips repack and the per-expert tables (byte-compare to a Python dump).
- The n-gram row decode (host C) — a real pack row compared to Python.
- The `weightBytes` arithmetic and the gate's admit/refuse decision.

---

## 6. Risks and honest sizing

- **Residency is the gate, not the kernels.** If the gate is left counting the
  mmapped 36 GiB table, the pack is refused-or-knife-edge regardless of how good
  the kernels are; the Python line serves it only because it mmaps. The Zig
  loader *already* mmaps it (§4.2) — the fix is to make the gate agree.
- **The pack mixes bit widths (4/5/6).** A port that assumes one width per
  checkpoint will be wrong everywhere except the experts. Read the width from the
  trellis shape, per tensor, as `format.py:bits_of` does.
- **Two real rewrites, not vendoring:** the Triton prompt GEMM (`prefill.py`) and
  the `exl3_mm.py` Triton kernels have no CUDA C++ equivalent — ~400–700 new
  lines, and the only genuinely new arithmetic.
- **The WC/mid-M kernel is not in the reference head** (`origin/perf/exl3-midm-wc`
  only). Port it later if the verify-window lane band is worth chasing.
- **A `mul1_multiplier` of `2212286765` (= `0x83DCD12D`) is the codebook marker
  value, not a per-tensor constant** — the decode constants live in `decode.cuh`
  (`:84-110`), not in the pack.
- **Nothing here reintroduces Python at serve time.** The vendored code is CUDA
  C++ compiled to a fatbin; the orchestration is Zig; the paging is `mmap`/`mlock`
  through `std.posix`. No torch, no Triton, no Python interpreter in the served
  engine.

---

### Appendix: file map

- Format & reference decode: `upstream/python-0.6:src/tensorfold/cuda/exl3/format.py`
- Tile decoder (pure CUDA): `…/exl3/decode.cuh`
- Dense kernels (pure) + launchers (torch): `…/exl3/linear.cu`, `linear.cpp`, `linear.py`
- Expert kernels (pure) + launchers (torch): `…/exl3/experts_grouped.cuh`,
  `experts.cu`, `experts_cb{0,1,2}.cu`, `experts.cpp`, `experts.py`
- Prompt GEMM (Triton): `…/exl3/prefill.py`; family Triton: `…/families/qwen4_exp/cuda/exl3_mm.py`
- Python n-gram paging: `…/cuda/ngram_pages.py`, `…/families/qwen4_exp/host_table.py`,
  `…/families/qwen4_exp/cuda/exl3_pack.py`
- Zig kernel registry: `zig/build/cuda.zig`, `zig/src/cuda/kernels.zig`
- Zig family/format: `zig/src/families/flashnext/cuda_native.zig`, `zig/src/core/engine_api.zig`
- Zig config: `zig/src/families/flashnext/cuda_config.zig`
- Zig weight loader: `zig/src/families/flashnext/cuda_weights.zig`
- Zig n-gram: `zig/src/families/flashnext/cuda_ngram.zig`
- Zig residency: `zig/src/native/cuda.zig`, `zig/src/native/cuda_memory.zig`
