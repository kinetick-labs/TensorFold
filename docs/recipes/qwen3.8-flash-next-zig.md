# Qwen3.8 Flash-Next, Zig engine on one GB10

This recipe serves **Qwen3.8 Flash-Next** (`qwen4_exp`) from its **EXL3 4.05 bpw** pack on **one NVIDIA
GB10** (DGX Spark, 128 GB unified memory) with the **Zig engine**: the native `tensorfold-native`
binary, not the Python runtime. It is the Zig counterpart of the public Python recipe
[`sxuff/qwen38-flash-next-dgx-spark`](https://github.com/sxuff/qwen38-flash-next-dgx-spark): same
family, same checkpoint, a different engine with the same mechanisms under Zig names.

Default operating point: **262,144-token window, bf16 KV, two streams, CUDA graphs on**. The draft cap is 15.
A round asks about 3–3.6 drafts from acceptance rates and holds about 2; the product stop at -0.4 only truncates that chain.

**Scope.** Text only. The Zig engine has no vision turn: it does not load the checkpoint's vision
tower and exposes no image or video input. The vision weights (`vision_k6.safetensors` in the pack, a
BF16 tower, image caching and the `--vision` flag) belong to the Python recipe and the MLX path. This
recipe also covers **one GPU**; the engine carries a two-rank link for this family (unlike sxuff's
single-GPU profile), but this page measures and qualifies the one-GPU configuration only.

## The recipe

```text
Language weights: turboderp/Qwen3.8-Flash-Next-exl3, revision 55a732e0c4c3d4614bc42b68493bb930d9b02c0a
Quantization:     4.05bpw_h6_ng6
Engine:           Zig (tensorfold-native), Zig 0.17.0
Hardware:         one GB10, 128 GB unified memory
Context:          262,144 tokens
MTP:              cap 15 (TF_FLASHNEXT_DEPTH). Ask ~3–3.6 from acceptance rates, held ~2; -0.4 only truncates (TF_FLASHNEXT_CONFIDENCE)
KV cache:         bf16 (default)
Streams:          2 (--parallel 2)
Experts window:   2048 rows (TF_FLASHNEXT_X3_MOE_WINDOW)
N-gram table:     rows gathered per prompt chunk, no prefetch stage (TF_FLASHNEXT_PREFILL_ROWS, default 2048)
CUDA graphs:      on by default (TF_FLASHNEXT_GRAPH_LOG counts them)
Vision:           not supported (text only)
API:              OpenAI-compatible, bound to 127.0.0.1
```

## 1. Weights

The language checkpoint is turboderp's EXL3 pack:

- Repository: `turboderp/Qwen3.8-Flash-Next-exl3`
- Branch: `4.05bpw_h6_ng6`
- Pinned revision: `55a732e0c4c3d4614bc42b68493bb930d9b02c0a` (the revision sxuff's
  `manifests/exl3-405.json` pins; the pack's `config.json` reports `model_type: qwen4_exp`)
- Local path used here: `/home/jct-spark/models/exl3-405` (about 101 GiB on disk)

Download it at that revision, or reuse an existing copy:

```bash
python -c "from huggingface_hub import snapshot_download as d; d('turboderp/Qwen3.8-Flash-Next-exl3', revision='55a732e0c4c3d4614bc42b68493bb930d9b02c0a', local_dir='/home/jct-spark/models/exl3-405')"
```

The pack carries nine `model-0000N-of-00009.safetensors` shards, an `ngram_embedding.safetensors`
(about 39 GB) that the engine **memory-maps** rather than loads resident, and an MTP head inside the
shards (no separate language drafter). It also ships a vision tower the Zig engine ignores. The
checkpoint's index, `tokenizer.json`, `chat_template.jinja` and `config.json` are read from the same
folder; keep the weights outside the source checkout.

## 2. Build

The Zig toolchain is **0.17.0**, pinned in `.zig-version`. On this host it is
`/home/jct-spark/zig-0.17.0/zig`. **`build.zig` sits at the root of the clone** (not under `zig/`):
every command below runs from the repository root.

### Development build

```bash
zig build native        # -> zig-out/native/bin/tensorfold-native
```

`tensorfold-native serve MODEL [flags]` is the server; the release archives also expose it as
`tensorfold`.

### Release distribution

`zig build dist` writes `macos-arm64`, `linux-x86_64` and `linux-aarch64` archives to `zig-out/dist/`.
It is deliberately strict. Run it from the clone root with:

- `-Dversion=MAJOR.MINOR.PATCH` — required; `dist` refuses without it.
- `-Ddist-fatbins=DIR` — required for a CUDA archive. Build the flat directory with:
  ```bash
  zig build fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=86,120,121
  ```
  `-Dsm=86,120,121` is required for the release fatbins: **sm_86** (Ampere), **sm_120** (RTX 50 /
  RTX PRO Blackwell) and **sm_121** (GB10). `dist` refuses any fatbin missing one of the three
  images. The default `-Dsm` is `121` alone, which would fail the release check.
- `-Dxgrammar=DIR` — required (structured output's `libtfgrammar`). It must be an xgrammar **v0.2.8**
  git checkout at that tag with submodules:
  ```bash
  git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar
  ```
- `-Ddist-macos-sdk=DIR` — required on a non-Mac builder. The SDK only builds the macOS archive, but
  `dist` aborts before building **any** archive when `xcrun` is unavailable and no override is given
  (`zig/build/dist.zig:55`); on Linux the whole run fails with *"the macOS archive requires a macOS
  SDK; use a Mac builder or -Ddist-macos-sdk=DIR"*.
- `-Ddist-cuda-aot=DIR` — the captured Triton set (`sm<capability>/aot.json` + `cubins/`), installed
  under `share/tensorfold/cuda/`. Optional to the packaging step; **required at runtime for this
  EXL3 pack** (see below). Without it, serve with `TENSORFOLD_CUDA_KERNELS` pointed at a qualified
  capture directory.

```bash
zig build dist \
  -Dversion=MAJOR.MINOR.PATCH \
  -Ddist-fatbins=/absolute/path/to/fatbins \
  -Dxgrammar=/absolute/path/to/xgrammar \
  -Ddist-macos-sdk=/absolute/path/to/MacOSX.sdk \
  -Ddist-cuda-aot=/absolute/path/to/capture
```

#### The AOT set must contain `_ple_rows`

The pack's n-gram table stores its rows in ExLlamaV3's mul1 trellis, not as EXL3 tiles. The engine
gathers a prompt chunk's rows and decodes them **on the device** with the captured Triton kernel
`_ple_rows` (`zig/src/families/flashnext/cuda_mia_prompt.zig:80`). If the capture shipped in the package
has no `_ple_rows` variant, the launch site's lookup fails and the engine stops with
`error.MissingTritonVariant` (`zig/src/cuda/aot.zig:134`) — the load/serve of the pack dies before a
token is produced. Check the packaged archive:

```bash
grep -c _ple_rows <pkg>/share/tensorfold/cuda/sm121/aot.json   # must be non-zero
```

For GB10 (`sm_121`) the relevant directory is `share/tensorfold/cuda/sm121/`.

## 3. Serving

sxuff's profile is a set of service settings; in the Zig engine each maps to a variable or flag.
**This recipe serves the engine's own defaults.** `TF_FLASHNEXT_DEPTH` unset is 15, and that number
is the cap (`most_drafts`), not the drafts asked each round. `depth()` in `core/lanes/depth.zig`
picks the ask from acceptance rates. `Config.init` stores the product rule as `head_stops`, but
`Engine.init` (`core/lanes/engine.zig`) never copies `head_stops` or `head_stops_most` onto the
rule, and nothing sets `Rule.live`, so `stopsNow()` stays false. The served `-0.4` product rule
runs in the head (`cuda_mia_lanes.zig` `confidenceNow`, `cuda_decode.zig` `Stop.take`) and only
truncates the chain already asked. On the 50.4 tok/s serve the ask is about 3–3.6 drafts and the
head holds about 2. Forcing `head_stops` so every ask is the cap costs 7–17%.

The table is how sxuff's depth 6 / 0.70 profile maps onto those variables. Depth and confidence
are read from the
environment, not from the server flags — the native CUDA Flash Next path builds its options from
`TF_FLASHNEXT_DEPTH` and `TF_FLASHNEXT_CONFIDENCE`
(`zig/src/families/flashnext/cuda_native.zig:91`, `cuda_engine.zig:62`); the parsed `--mtp-drafts` /
`--mtp-confidence` server flags are not wired to this family.

| sxuff (Python 0.6.5) | Zig engine | Read in |
| --- | --- | --- |
| `depth=6` | `TF_FLASHNEXT_DEPTH=6` (unset is the cap 15, not a fixed ask of 15) | `cuda_native.zig:91` |
| `confidence=0.70` | `TF_FLASHNEXT_CONFIDENCE=0.70` (per-draft rule; served default is the running-product rule at `-0.4`) | `cuda_engine.zig:62` |
| `max_len=262144` | `--context 262144` (default) | `server/cli.zig` |
| `tp=1` | one GPU (default) | — |
| `streams=2` | `--parallel 2` | `server/cli.zig:300` |
| `kv_dtype="bf16"` | `--kv-dtype bf16` (default) | `server/cli.zig:300`, `cuda_state.zig:19` |
| `graphs=True` | on by default; `TF_FLASHNEXT_GRAPH_LOG=1` counts them | `cuda_engine.zig:73`, `cuda_engine.zig:463` |
| `prefetch=False` (n-gram demand-paged) | rows gathered per prompt chunk, no prefetch stage; `TF_FLASHNEXT_PREFILL_ROWS` sets the rows (default 2048 on one GPU) | `cuda_engine.zig:273` |
| `vision=True` | **not supported** | — |

Confidence detail: the checkpoint's packed profile is the MTP-6 / 0.70 rule. `cuda_engine.zig:40`
keeps that value as `default_confidence = 0.70`, but the **served** default is the running-product
stop rule (`served_confidence = -0.4`, `cuda_engine.zig:47`). A positive `TF_FLASHNEXT_CONFIDENCE`
selects Python's per-draft rule, so `TF_FLASHNEXT_CONFIDENCE=0.70` reproduces sxuff's threshold;
leaving it unset serves the product rule instead. `default_depth = 6` (`cuda_engine.zig:39`) is the
options-struct default; `cuda_native.zig:91` passes 15 when `TF_FLASHNEXT_DEPTH` is unset. That 15
is the most drafts a round may ask. The acceptance-rate rule asks about 3–3.6, and the product
stop shortens what the head keeps.

Two further defaults worth knowing: `TF_FLASHNEXT_PRODUCT_STREAMS=2` switches the head's stop
(the product rule while at most two lanes are drafting, then `wide_confidence` 0.5; a different
concept from `--parallel 2`; `cuda_mia_lanes.zig` `confidenceNow`, `cuda_engine.zig:53`). It does not
make `depth()` ask the cap. `TF_FLASHNEXT_X3_MOE_WINDOW=2048` is the grouped-experts window
(`cuda_forward.zig:335`).

The command below is the served operating point behind the **50.4 tok/s** decode. Unset variables
resolve to the same values. Depth 15 is the cap.

```bash
export TF_FLASHNEXT_DEPTH=15
export TF_FLASHNEXT_CONFIDENCE=-0.4
export TF_FLASHNEXT_PRODUCT_STREAMS=2
export TF_FLASHNEXT_X3_MOE_WINDOW=2048
# optional: confirm the decode/verify graphs are being captured
# export TF_FLASHNEXT_GRAPH_LOG=1

zig-out/native/bin/tensorfold-native serve /home/jct-spark/models/exl3-405 \
  --host 127.0.0.1 --port 8001 --name bench \
  --context 262144 --parallel 2 --kv-dtype bf16
```

The endpoint is bound to `127.0.0.1:8001`; this recipe does not publish an unauthenticated API. The
server refuses at startup when two streams at a 262,144-token window do not fit the memory budget
after the weights; lower `--parallel` or `--context` then. Two streams mirror sxuff's profile (there
they are required for vision, which this engine lacks); one stream also serves text. The
**peak-memory** figure for this configuration remains to be measured.

## 4. Measured figures

Decode and draft-acceptance figures below were taken on one GB10 (sm_121) with the checkpoint and
the serve command above, through the Zig server. Treat each as a band, not a specification. The
Fibonacci prompt is `tools/bench_openai.py`'s `fibonacci-raw`: "Write a short Python function that
computes the Fibonacci sequence and explain it." Decode tok/s is `(completion tokens - 1) / (last
token time - first token time)`, the median of three repetitions, temperature 0. One request is one
live stream, so the head's stop stays the running product (`p1 * ... * pj >= 0.4`) rather than
`wide_confidence` 0.5. That rule truncates the chain. It does not choose how many drafts to ask.

| Metric | Value | Method | Reserve |
| --- | ---: | --- | --- |
| Decode, Fibonacci prompt | 50.4 tok/s | cap 15, acceptance-rate ask (~3–3.6), product stop -0.4 holds ~2, experts window 2048, temperature 0, median of three | One GB10, `--context 262144`, `--parallel 2`, bf16 KV, graphs on |
| Decode, chat prompt | 42.7 tok/s | same serve and timing | as above |
| Prefill | 770–780 tok/s | ~14,400-token prompt with `TF_FLASHNEXT_PREFILL_ROWS=4096` | Chunk rows 4096, above the one-GPU default 2048 |
| Draft chain | ~2.25 tokens/round | same serve, draft-chain log | acceptance by level 0.66 / 0.40 / 0.33 / 0.09 |
| Served output after the rebase | bit-identical | `token_sha afccd937e6ac` matched before and after the rebase | One prompt |

A round is about 48 ms. Verify is about 36 ms at 1 row, 46 ms at 3, and 55 ms at 4. A draft step
is about 2.9 ms. Host gaps around the phases are about 0.01 ms. The depth rule asks about 3–3.6
drafts and the product stop holds about 2, which is the ~2.25 tokens a round. Acceptance by level
is 0.66, 0.40, 0.33, 0.09. An earlier pass on this same serve reported 2.22 tokens a round and
56.7% acceptance as one figure. A window can hold `depth + 1` = 16 rows; a round does not fill it.
Forcing `head_stops` so every ask is that cap costs 7–17%.

**Not comparable to sxuff's 75.68 tok/s.** That figure is the Python recipe's short-prompt decode on
**40,960-token context, text only, one stream**; this page's operating point is **262,144** and two
streams. The two numbers do not describe the same deployment, and no like-for-like A/B between the
Python and Zig engines on this pack exists here. Other quantities remain to be measured: peak memory,
time to first token, prefill at other lengths, and multi-stream aggregate throughput.

## 5. Differences from the Python recipe

- **Engine.** The Python recipe runs TensorFold 0.6.5 (the `src/` runtime) with a vision-prefix patch.
  This recipe runs the Zig engine (`tensorfold-native`). Same family, same checkpoint; the arithmetic
  is the engine's own, so tokens are exact within this engine (drafted against serial) rather than
  against Python.
- **No vision.** The Python recipe's headline is native vision, image-aware prefix caching and a
  verified BF16 tower; the Zig engine has none of these. Its scope is text alone.
- **Stricter distribution.** The Python recipe builds a container (`scripts/build_tensorfold.sh`).
  The Zig distribution is a set of native archives and is strict about its inputs: an xgrammar
  v0.2.8 checkout, a macOS SDK, release fatbins for `sm_86,sm_120,sm_121`, and a captured AOT set
  that must contain `_ple_rows` for this pack.
- **Configuration surface.** sxuff's `scripts/serve.py` profile is a Python dict; the Zig engine
  takes the same intent through environment variables (`TF_FLASHNEXT_DEPTH`, `TF_FLASHNEXT_CONFIDENCE`)
  and server flags (`--context`, `--parallel`, `--kv-dtype`).

## Licenses

TensorFold is Apache-2.0 licensed; the xgrammar shim is xgrammar v0.2.8 (Apache-2.0). The model
weights are governed separately by the [Qwen Community License](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3).
Review the model license before downloading or deploying them.
