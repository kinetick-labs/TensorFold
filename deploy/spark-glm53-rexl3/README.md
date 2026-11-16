# Spark GLM-5.3 serve defaults: SAGE REXL3, 2 × 262,144, paged KV, vision, DFlash2

Serves [vcruz305](https://huggingface.co/vcruz305)'s SAGE REXL3 2.48 bpw pack of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) on one DGX Spark, port 8094,
container `glm53`, from a ReleaseFast build of `feat/glm53-rexl3`. This is not prod: Qwen on :8083
is unchanged.

The pack is MIT (cite the base model). Drafts are [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).

## Config

- `TF_KV_PAGES=on`, `--parallel 2`, `--context 262144`, `TENSORFOLD_MEMORY_RESERVE_GIB=5`
- `--vision`, `--drafter` DFlash2 (`~/models/glm53-dflash2`)
- Two streams share a 3.20 GiB page pool (514 pages of 512 rows), enough for one full 262,144-token
  window; a second stream uses pages as they free. Two reserved whole 262k windows do not fit
  beside the weights, the 1.3 GiB vision tower and the draft.

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
The 0.43 GiB prompt cache cannot keep a long state by copying; `TF_KV_SHARE=n` keeps a prompt's
rows as the pool's pages instead (tested separately).
