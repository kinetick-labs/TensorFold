# Spark GLM-5.3 serve defaults: SAGE REXL3, 2 × 262,144, paged KV, vision, DFlash2, conversation reuse

Serves [vcruz305](https://huggingface.co/vcruz305)'s SAGE REXL3 2.48 bpw pack of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on one DGX Spark, port 8094,
container `glm53`, from a ReleaseFast build of `feat/glm53-rexl3`. This is not prod: Qwen on :8083
is unchanged.

The pack is MIT (cite the base model). Drafts are [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

## Config

- `TF_KV_PAGES=on`, `--parallel 2`, `--context 262144`, `TENSORFOLD_MEMORY_RESERVE_GIB=5`
- `--vision`, `--drafter` DFlash2 (`~/models/glm53-dflash2`)
- EXL3 prompt experts run on FP8 (build default since 9939ace88; `TF_GLM_X3_FP8=0` gives f16 back).
- `TF_KV_SHARE=2`, `TF_GLM_KV_LIVE_HALF=1`, `TF_GLM_HOST_CACHE=1`, `--prompt-cache-gib 1`: conversation
  reuse. A finished prompt keeps its rows as the pool's pages (up to 2 kept states); the KDA snapshot
  of each kept state lives in host RAM, outside the pool, and carries only the live half of the KDA
  state (302 MiB instead of 438 MiB, no precision loss: warm outputs stay bit-identical to cold). A next turn only prefills its new tokens (from the
  last 2048-row mark).
- A start refused with "budget fits no stream" (the KV budget can come up ~0.1 GiB short right after
  another server stops) is retried by serve.sh up to 3 times, 20 s apart.
- `TENSORFOLD_PREFILL_POLICY=fcfs`: between 2048-row prompt passes a short prompt goes first, so it
  does not wait for a long prefill to finish.
- Two streams share a 3.45 GiB page pool (554 pages of 512 rows), enough for one full 262,144-token
  window; a second stream uses pages as they free. Two reserved whole 262k windows do not fit
  beside the weights, the 1.3 GiB vision tower and the draft. Kept states are evicted when a live
  stream needs their pages: one long conversation (or two of ~97k, measured) is reused, two concurrent
  ~247k conversations are not (both next turns prefill cold).

## Install

```
# ReleaseFast native build of this branch, then:
BIN=~/tf-glm53/zig-out/native DRY_RUN=1 bash deploy/spark-glm53-rexl3/serve.sh   # print the command
BIN=~/tf-glm53/zig-out/native bash deploy/spark-glm53-rexl3/serve.sh             # start (replaces glm53 only)
```

`LIBS` defaults to `~/tf-prod-df2g/libs` (the Spark's CUDA/vision runtime libs). The script never
touches the Qwen unit or container.

## Bench (Spark, 16 Nov 2026, window 25, c18970d85, DFlash2 on)

MemAvailable stayed at or above 4.7 GiB (low 4741 MiB during a 247k+62k pair). Needles were found.

| Workload | prompt tokens | TTFT | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1 short prose | 25 | 0.2 s | — | 30.7 |
| 2 short prose | 25 + 25 | 0.2 / 0.5 s | — | 27.0 / 27.6 |
| one 64k fill | 70,320 | 71.5 s | 984 | 31.5 |
| 2 × ~130k fills | 129,541 + 134,441 | 133.5 / 277.9 s | 971 / 484 | 30.8 / 28.9 |
| 247k + 62k fills | 247,071 + 62,521 | 260.8 / 331.5 s | 947 / 189 | 27.8 / 25.5 |

A 277,072-token mix was refused (`context_length_exceeded`). Video is not served (no libtfvideo).

## Conversation reuse and short-first prefill (16 Nov 2026, windows 26–28, 45199be92 build)

Same build and config as above plus `TF_KV_SHARE=2 TF_GLM_HOST_CACHE=1 --prompt-cache-gib 1`
(fcfs in window 28). Needles were found in every run. Warm turn-2 outputs are bit-identical to a
cold prefill of the same prompt (sha 1852594d29e5 at 120k, 981cc5c6fe29 at 247k).

| Workload | prompt tokens | cached | TTFT |
|---|---|---|---|
| turn 1, 120k conversation | 119,728 | 0 | 122.9 s |
| turn 2, same conversation | 119,755 | 118,784 | 1.23 s |
| turn 1, 247k conversation | 247,058 | 0 | 259.1 s |
| turn 2, same conversation | 247,085 | 245,760 | 1.65 s |
| turn 2 after a 197k fill evicted it | 247,085 | 0 | 260.6 s |
| 2 conversations of ~97k, alternating, turn 2 | 97,303 / 97,301 | 96,256 each | 1.29 / 1.29 s |
| short prompt during a 247k prefill (fcfs) | 25 (+247,056) | 0 | 1.01 s (long 259.3 s) |
| short prompt during a 247k prefill (default policy) | 25 (+247,059) | 0 | 1.55 s (long 262.5 s) |
| 2 concurrent 247k fills, turn 1 | 247,058 × 2 | 0 | 255.8 / 512.2 s |
| same 2 conversations, concurrent turn 2 | 247,085 × 2 | 0 | 258.1 / 515.6 s |
| 247k + 60k concurrent, turn 1 | 247,057 + 60,038 | 0 | 319.6 / 61.5 s |
| same, concurrent turn 2 | 247,083 + 60,065 | 0 | 321.0 / 61.7 s |

While a long prefill runs, the short stream decodes at ~1.2 tok/s until the long one finishes.
Memory (window 28, 2 concurrent 247k fills with kept states): MemAvailable low 4360 MiB, the
1 GiB watchdog never tripped (window 27 low: 4156 MiB).

## FP8 prompt experts, live-half kept states, dropped experiments (16 Nov 2026, windows 29–30)

Build 72d1c64c2 (FP8 default on, KDA f_b/g_b pair launch), same config as above.

- Prefill with FP8 experts: 968 tok/s at 2.9k (968/968/966), 1009 tok/s at 14k (1009/1005);
  ~960 / ~980 before. ~1000–1026 tok/s on 120k–247k fills.
- Decode (DFlash2): prose 29.9, code 49.1, math 54.3, structured 63.2, French 33.6 tok/s.
  FP8 changed no answer on the 64-question bench but adds 9–10 % perplexity on long prose and one
  known arithmetic drift in the coherence set.
- Live-half kept states: 302 MiB a snapshot instead of 438 MiB. With `TF_KV_SHARE=4`, four ~28k
  conversations all resumed on turn 2 (cached 26,624–28,672, TTFT 0.5–1.4 s vs 27.8–28.9 s cold,
  outputs identical to cold); a 120k turn 2 resumed 118,784 tokens in 1.20 s (turn 1 116.7 s).
- Memory with `TF_KV_SHARE=4`: MemAvailable low 3971 MiB (4 × 28k), 3528 MiB (a 120k turn pair, then
  2 concurrent 247k fills); the 1 GiB watchdog never tripped. The default stays at 2 kept states
  for margin; `TF_KV_SHARE=4` is opt-in. Two concurrent ~247k conversations are still not reused
  (the pool fits one 262k window).

Measured and not kept:

- `TF_GLM_KDA_FG_PAIR` (default on, bit-identical): no measurable decode gain; the whole KDA decode
  chain is ~5 % of decode GPU time (EXL3 experts ~43 %, linears ~30 %).
- `TF_GLM_DRAFT_ADAPT=1` (adaptive DFlash2 draft depth, default off): no gain (prose 29.8, fr 33.2).
- Power-of-two ladder of kept intermediate states: no TTFT gain on agent replays (186.5 s vs 183.7 s)
  and it took the kept-state slots, so conversation turn 2 lost its reuse. Not committed.
- Finer resume grid for short (4.5–6k) agent prompts: GLM resumes only at a cold 2048-row cut, so a
  finer store grid cannot help, and `TF_GLM_REQUEST_STARTS=1` is refused with `TF_KV_SHARE`. Reuse
  between cuts would need the engine to resume off the cold grid (losing warm == cold identity).
- `TF_GLM_CHUNK_SOLO=4096` (4096-row prompt passes): does not boot at 2 × 262,144 with vision and
  drafts (0.89 GiB left for a 3.19 GiB window); usable only with a smaller context.
