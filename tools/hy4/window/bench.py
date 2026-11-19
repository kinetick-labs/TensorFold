# bench.py PORT: streamed chats to a Hy4 server, thinking off: TTFT -> prefill tok/s, then decode tok/s over the reply.
import json, sys, time, urllib.request
p = sys.argv[1]; U = f"http://127.0.0.1:{p}/v1/chat/completions"
m = json.load(urllib.request.urlopen(f"http://127.0.0.1:{p}/v1/models", timeout=10))["data"][0]["id"]
filler = "The history of the river town is long and quiet; merchants came and went with the seasons. "
def run(label, words, n):
    prompt = (filler * (words // 16 + 1)) + "\nWrite a long story about this town."
    b = {"model": m, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; last = None; chunks = 0; usage = {}
    try:
        with urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                d = json.loads(line[5:])
                if d.get("usage"): usage = d["usage"]
                for c in d.get("choices", []):
                    if (c.get("delta") or {}).get("content"):
                        now = time.time(); first = first or now; last = now; chunks += 1
    except Exception as e:
        print(f"{label}: error {e}"); return
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    ttft = (first or time.time()) - t0
    dec = (ct - 1) / (last - first) if first and last and last > first and ct > 1 else 0
    print(f"{label}: prompt {pt} tok, TTFT {ttft:.2f} s ({pt / ttft:.0f} tok/s prefill), {ct} tok decoded at {dec:.2f} tok/s")
run("short", 20, 128)
run("2k", 1500, 128)
run("8k", 6000, 64)
