# Spark prod recipe: FP4-Local on the mia engine with DFlash2, strict tools and srf sliced prefill

Serves Qwen3.8 Flash Next FP4-Local (nvidia, rev b9ce18a6) on one DGX Spark, port 8083, container
`qwen38-fp4local`, under the user unit `qwen38-flash-next-tensorfold.service`.

## Config
- fp8 KV, `--prompt-cache-gib 12`, `--parallel 3`, `--context 262144`, `--vision-helper`
- DFlash2 drafter `~/models/qwen38-dflash2`: 7 drafts, cut 0.15, wide cut 0.5
- strict tools / `tool_choice: required` / `json_schema` through the grammar hooks (libtfgrammar), after
  Jay Leaton's (jayleaton) upstream strict-tool work
- `TF_FLASHNEXT_FILL=1` layer-sliced prefill (after MiaAI-Lab), `TENSORFOLD_PREFILL_POLICY=srf`,
  `TF_FLASHNEXT_FILL_LAYERS_BUSY=1`, `TF_FLASHNEXT_FILL_ROWS_BUSY=1024`, `TF_FLASHNEXT_FILL_ROWS_HOLD_MS` 3000 (default)

## Install
```
ROOT=~/tf-prod-df2g
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=ReleaseFast   # at the pushed commit
cp zig-out/native/bin/tensorfold-native $ROOT/bin/; git rev-parse HEAD > $ROOT/COMMIT
cp libtfgrammar.so libturbojpeg.so* libwebpdecoder.so* $ROOT/libs/
cp deploy/spark-fp4-df2g/serve.sh $ROOT/
cp deploy/spark-fp4-df2g/zz-df2g.conf ~/.config/systemd/user/qwen38-flash-next-tensorfold.service.d/
systemctl --user daemon-reload && systemctl --user restart qwen38-flash-next-tensorfold.service
```
Rollback: remove `zz-df2g.conf`, daemon-reload, restart (the previous drop-in is left in place).

## Bench (Spark, fp8 KV, p3, 262k, 3 distinct agents x 4 turns, 256 tokens a reply)
| config | turn-1 avg | A/B/C turn-1 | later turns avg/max | decode avg/min tok/s | cache cap | solo prefill 14k | hits |
|---|---|---|---|---|---|---|---|
| fp8 base, no slicing | 38.6 s | 21.4/32.8/61.7 | 0.74/1.22 s | 22.5/4.7 | 11753 MiB | 2328 tok/s | 9/12 |
| sliced, 3 prompts interleaved (busy 4) | 65.7 s | 70.1/56.6/70.2 | 0.83/1.16 s | 27.8/20.1 | 11553 MiB | 2313 | 9/12 |
| **this recipe** (srf, busy 1, rows-busy 1024, hold 3 s) | **64.0 s** | 22.4/58.5/111.0 | **1.06/1.34 s** | 44.1/**31.4** | 11521 MiB | **2313** | 9/12 |
DFlash2 single-stream code decode ~99-114 tok/s (MTP ~73). TC2 strict-tool/json checks 7/7.

Exactness caveat: verified on outputs only (greedy, temperature 0): all 12 multi-agent reply hashes and the
single-stream hashes identical to the fp8 base. Smaller chunks change the chunk cut, which the engine notes can
change bits; logits were not compared bit for bit.
