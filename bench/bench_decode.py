#!/usr/bin/env python3
"""Single-stream decode benchmark for an OpenAI-compatible Ling-3.2-air server (stdlib only).

One request at a time. For each target prompt length it builds a unique prompt from public-domain
novels (corpus/*.txt), asks for a long summary (thinking off, temperature 0, fixed max_tokens) and
reports prompt tokens, TTFT, prefill tok/s and decode tok/s. If the server exposes Prometheus
/metrics with speculative-decoding counters (vLLM, SGLang --enable-metrics) it also prints the
per-request MTP acceptance.

  python3 bench_decode.py --base http://localhost:8140 --model ling-3.2-air --lengths 1000 32000 \
      --out results/decode.json --label "mtp on"

Not a load test: never run it concurrently.
"""
import argparse
import glob
import json
import os
import re
import time
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))


def load_corpus():
    texts = []
    for p in sorted(glob.glob(os.path.join(HERE, "corpus", "*.txt"))):
        t = open(p, encoding="utf-8", errors="ignore").read()
        # drop the Project Gutenberg header/footer
        s = t.find("*** START OF")
        e = t.find("*** END OF")
        if s != -1:
            t = t[t.find("\n", s) + 1:]
        if e != -1:
            t = t[:e]
        texts.append(re.sub(r"\n{3,}", "\n\n", t).strip())
    return "\n\n".join(texts)


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read().decode()


def spec_metrics(base):
    """Return {name: value} for every metric line mentioning spec/accept/draft (sums over labels)."""
    try:
        txt = get(base, "/metrics")
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        low = line.lower()
        if not any(k in low for k in ("spec", "accept", "draft")):
            continue
        try:
            name_lbl, val = line.rsplit(" ", 1)
            name = name_lbl.split("{")[0]
            if name.endswith("_created"):
                continue
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def stream_chat(base, model, messages, max_tokens, extra=None, timeout=3600):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if extra:
        body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = t_last = None
    usage = None
    text = []
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks += 1
                    text.append(piece)
    n = (usage or {}).get("completion_tokens", 0)
    p = (usage or {}).get("prompt_tokens", 0)
    ttft = (t_first - t0) if t_first else None
    dec = (n - 1) / (t_last - t_first) if (t_first and t_last and t_last > t_first and n > 1) else None
    return dict(prompt_tokens=p, completion_tokens=n, ttft_s=ttft,
                prefill_tok_s=(p / ttft) if ttft else None, decode_tok_s=dec,
                wall_s=time.time() - t0, chunks=chunks, text="".join(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140")
    ap.add_argument("--model", default="ling-3.2-air")
    ap.add_argument("--lengths", type=int, nargs="+", default=[1000, 32000])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--chars-per-token", type=float, default=0.0, help="0 = calibrate with one request")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    corpus = load_corpus()
    ask = ("\n\n---\nWrite a detailed, multi-paragraph summary of the text above: the people, what happens, "
           "and the tone. Aim for at least 500 words.")

    cpt = a.chars_per_token
    if not cpt:
        sample = corpus[:20000]
        r = stream_chat(a.base, a.model, [{"role": "user", "content": sample + ask}], 1)
        cpt = len(sample + ask) / max(1, r["prompt_tokens"])
        print(f"calibration: {r['prompt_tokens']} tokens for {len(sample + ask)} chars -> {cpt:.3f} chars/token")

    # warmup (short)
    stream_chat(a.base, a.model, [{"role": "user", "content": "Say hello in five words."}], 16)

    results = []
    for L in a.lengths:
        for rep in range(a.repeats if L <= 40000 else 1):
            nonce = f"[run {uuid.uuid4().hex[:12]}]\n"
            nchars = int(max(0, L - 60) * cpt)
            off = (hash(nonce) % max(1, len(corpus) - nchars - 1)) if len(corpus) > nchars else 0
            doc = corpus[off:off + nchars]
            if len(doc) < nchars:  # wrap around for very long prompts
                doc = (corpus * (nchars // len(corpus) + 2))[:nchars]
            m0 = spec_metrics(a.base)
            r = stream_chat(a.base, a.model, [{"role": "user", "content": nonce + doc + ask}], a.max_tokens)
            m1 = spec_metrics(a.base)
            delta = {k: m1[k] - m0.get(k, 0.0) for k in m1 if abs(m1[k] - m0.get(k, 0.0)) > 1e-9}
            r["spec_metrics_delta"] = delta
            # vLLM counters
            acc = delta.get("vllm:spec_decode_num_accepted_tokens_total")
            drf = delta.get("vllm:spec_decode_num_draft_tokens_total")
            nd = delta.get("vllm:spec_decode_num_drafts_total")
            if acc is not None and drf:
                r["mtp_acceptance_rate"] = acc / drf
                r["mtp_mean_accept_len"] = 1 + acc / nd if nd else None
            r["target_tokens"] = L
            r["label"] = a.label
            r["text_head"] = r.pop("text")[:300]
            results.append(r)
            extra = ""
            if "mtp_acceptance_rate" in r:
                extra = f"  MTP accept {r['mtp_acceptance_rate']*100:.1f}%  mean len {r['mtp_mean_accept_len']:.2f}"
            print(f"{L:>7} target | prompt {r['prompt_tokens']:>7} | out {r['completion_tokens']:>4} | "
                  f"TTFT {r['ttft_s']:7.2f} s | prefill {r['prefill_tok_s'] or 0:8.0f} tok/s | "
                  f"decode {r['decode_tok_s'] or 0:6.1f} tok/s{extra}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(dict(base=a.base, model=a.model, label=a.label, chars_per_token=cpt,
                       max_tokens=a.max_tokens, time=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                       results=results), open(a.out, "w"), indent=2)
        print("saved", a.out)


if __name__ == "__main__":
    main()
