#!/usr/bin/env python3
"""One very long prompt against the server, with a memory brake. Run it ON the Spark (it reads /proc/meminfo).

Builds a prompt from the novels in bench/corpus/ (Project Gutenberg pg1342, pg2701, pg98 as .txt), asks for the
first sentence of the text, and watches MemAvailable while the prompt is prefilled. If MemAvailable falls under
BRAKE_MIB (default 9216) it drops the connection, so vLLM aborts the request before the box runs out of memory.

  python3 bench/longprompt.py 880000                 # ~215K tokens with this tokenizer
  BRAKE_MIB=8192 python3 bench/longprompt.py 400000 --base http://localhost:8140
Prints one JSON line: ok, prompt_tokens, answer, seconds, min_mem_available_mib, braked.
"""
import argparse, http.client, json, os, threading, time, urllib.parse

ap = argparse.ArgumentParser()
ap.add_argument("chars", nargs="?", type=int, default=880000, help="prompt length in characters")
ap.add_argument("--base", default="http://localhost:8140")
ap.add_argument("--model", default="ling-3.2-air")
ap.add_argument("--corpus", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))
a = ap.parse_args()
BRAKE_MIB = int(os.environ.get("BRAKE_MIB", "9216"))

files = sorted(f for f in os.listdir(a.corpus) if f.endswith(".txt"))
if not files:
    raise SystemExit(f"no .txt files in {a.corpus}")
txt = ""
while len(txt) < a.chars:
    for f in files:
        txt += open(os.path.join(a.corpus, f), errors="ignore").read()
txt = txt[: a.chars]
q = "\n\nQuestion: In one sentence, what is the very first sentence of the text above?"
body = json.dumps({"model": a.model, "max_tokens": 64, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "user", "content": txt + q}]})

u = urllib.parse.urlparse(a.base)
conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=1800)
state = {"min": 10**9, "braked": False, "done": False}


def watch():
    while not state["done"]:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                m = int(line.split()[1]) // 1024
                state["min"] = min(state["min"], m)
                if m < BRAKE_MIB and not state["braked"]:
                    state["braked"] = True
                    try:
                        conn.sock.close()
                    except Exception:
                        pass
        time.sleep(0.2)


threading.Thread(target=watch, daemon=True).start()
t0 = time.time()
try:
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    d = json.loads(conn.getresponse().read())
    res = {"ok": True, "prompt_tokens": d["usage"]["prompt_tokens"],
           "answer": (d["choices"][0]["message"]["content"] or "")[:200]}
except Exception as e:
    res = {"ok": False, "error": str(e)[:200]}
state["done"] = True
res.update(seconds=round(time.time() - t0, 1), min_mem_available_mib=state["min"], braked=state["braked"])
print(json.dumps(res))
