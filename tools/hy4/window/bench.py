# bench.py PORT [--fill WORDS] [--parallel K] [--skip-base]: streamed chats to a Hy4 server, thinking off.
#   base runs: short / 2k / 8k prompts -> TTFT (prefill tok/s) and decode tok/s over the reply;
#   --fill WORDS: one long prompt (~1.3 tokens a word) -> TTFT and decode at that depth;
#   --parallel K: K short requests at once -> each stream's decode tok/s and the aggregate.
# Lines start with "BENCH" so window.log can be grepped.
import json, sys, time, threading, urllib.request
args = sys.argv[1:]
p = args[0]; U = f"http://127.0.0.1:{p}/v1/chat/completions"
def opt(name, default):
    return int(args[args.index(name) + 1]) if name in args else default
m = json.load(urllib.request.urlopen(f"http://127.0.0.1:{p}/v1/models", timeout=10))["data"][0]["id"]
filler = "The history of the river town is long and quiet; merchants came and went with the seasons. "

def run(label, words, n, quiet=False, out=None, salt=""):
    prompt = salt + (filler * (words // 16 + 1)) + "\nWrite a long story about this town."
    b = {"model": m, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; last = None; usage = {}
    try:
        with urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                d = json.loads(line[5:])
                if d.get("usage"): usage = d["usage"]
                for c in d.get("choices", []):
                    if (c.get("delta") or {}).get("content"):
                        now = time.time(); first = first or now; last = now
    except Exception as e:
        print(f"BENCH {label}: error {e}", flush=True); return None
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    ttft = (first or time.time()) - t0
    dec = (ct - 1) / (last - first) if first and last and last > first and ct > 1 else 0
    res = {"label": label, "prompt": pt, "ttft": ttft, "prefill": pt / ttft if ttft > 0 else 0, "tokens": ct, "decode": dec, "t0": t0, "first": first, "last": last}
    if not quiet:
        print(f"BENCH {label}: prompt {pt} tok, TTFT {ttft:.2f} s ({res['prefill']:.0f} tok/s prefill), {ct} tok decoded at {dec:.2f} tok/s", flush=True)
    if out is not None: out.append(res)
    return res

if "--skip-base" not in args:
    run("short", 20, 128)
    run("2k", 1500, 128)
    run("8k", 6000, 64)
fill = opt("--fill", 0)
if fill:
    run(f"fill {fill} words", fill, 64)
k = opt("--parallel", 0)
if k:
    res = []
    th = [threading.Thread(target=run, args=(f"stream {i}", 40, 128, True, res, f"[{i}] ")) for i in range(k)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    ok = [r for r in res if r and r["tokens"] > 1]
    total = sum(r["tokens"] for r in ok)
    # the span where every stream was decoding at once
    if ok:
        lo = max(r["first"] for r in ok); hi = min(r["last"] for r in ok)
        per = ", ".join(f"{r['decode']:.1f}" for r in ok)
        print(f"BENCH parallel {k}: {len(ok)}/{k} streams ok, {total} tokens in {wall:.1f} s wall ({total / wall:.1f} tok/s aggregate), per-stream decode tok/s [{per}], overlap {max(0, hi - lo):.1f} s", flush=True)
    else:
        print(f"BENCH parallel {k}: no stream finished", flush=True)
