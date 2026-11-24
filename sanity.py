#!/usr/bin/env python3
"""Quick sanity check for a Ling-3.2-air OpenAI-compatible server (stdlib only).

Runs three requests: a short answer with thinking off, the same with thinking on (default),
and one tool call. Prints the answers and a PASS/FAIL line for each.

  python3 sanity.py --base http://localhost:8140 --model ling-3.2-air
"""
import argparse
import json
import time
import urllib.request


def chat(base, payload, timeout=300):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140")
    ap.add_argument("--model", default="ling-3.2-air")
    a = ap.parse_args()
    ok = True

    # 1) thinking off
    d, dt = chat(a.base, {"model": a.model, "max_tokens": 64, "temperature": 0,
                          "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
                          "chat_template_kwargs": {"enable_thinking": False}})
    txt = (d["choices"][0]["message"].get("content") or "").strip()
    p = "391" in txt
    ok &= p
    print(f"[{'PASS' if p else 'FAIL'}] thinking off ({dt:.2f} s): {txt!r}")

    # 2) thinking on (the model's default); reasoning goes to reasoning_content via --reasoning-parser ling3
    d, dt = chat(a.base, {"model": a.model, "max_tokens": 2048, "temperature": 0,
                          "messages": [{"role": "user", "content": "A shop sells 3 shirts for $45. How much are 7 shirts? Final answer only."}]})
    m = d["choices"][0]["message"]
    txt = (m.get("content") or "").strip()
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    p = "105" in txt
    ok &= p
    print(f"[{'PASS' if p else 'FAIL'}] thinking on ({dt:.2f} s, {len(reasoning)} chars of reasoning): {txt[:120]!r}")

    # 3) tool call
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    d, dt = chat(a.base, {"model": a.model, "max_tokens": 512, "temperature": 0, "tools": tools, "tool_choice": "auto",
                          "messages": [{"role": "user", "content": "What's the weather in Atlanta right now?"}],
                          "chat_template_kwargs": {"enable_thinking": False}})
    calls = d["choices"][0]["message"].get("tool_calls") or []
    p = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "atlanta" in calls[0]["function"]["arguments"].lower()
    ok &= p
    print(f"[{'PASS' if p else 'FAIL'}] tool call ({dt:.2f} s): {calls[0]['function'] if calls else None}")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
