# smoke.py PORT IMAGE: one text and one image chat to a Hy4 server, thinking off (no_think) and on (high).
import base64, json, sys, time, urllib.request
p, img = sys.argv[1], sys.argv[2]
U = f"http://127.0.0.1:{p}/v1/chat/completions"
m = json.load(urllib.request.urlopen(f"http://127.0.0.1:{p}/v1/models", timeout=10))["data"][0]["id"]
def ask(content, think, n=128):
    try:
        return ask1(content, think, n)
    except Exception as e:
        body = e.read().decode()[:300] if hasattr(e, "read") else ""
        print("  error:", e, body)
        return {}
def ask1(content, think, n=128):
    b = {"model": m, "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600))
    u = r.get("usage", {}); dt = time.time() - t0
    print(f"  usage {u.get('prompt_tokens')} prompt + {u.get('completion_tokens')} completion tokens in {dt:.1f} s")
    return r["choices"][0]["message"]
ok = 0
t0 = time.time()
t = ask("What is the capital of France? One word.", False)
print(f"text ({time.time() - t0:.1f} s):", repr((t.get("content") or "")[:80])); ok += "paris" in (t.get("content") or "").lower()
data = base64.b64encode(open(img, "rb").read()).decode()
v = ask([{"type": "text", "text": "What color is the large square in this picture? One word."}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}], False)
print("image:", repr((v.get("content") or "")[:80])); ok += "red" in (v.get("content") or "").lower()
th = ask("What is 17 * 3?", True, 512)
print("think:", repr((th.get("reasoning_content") or "")[:60]), repr((th.get("content") or "")[:40])); ok += "51" in (th.get("content") or "")
print(f"smoke {ok}/3")
