# Ling-3.2-air (NVFP4) on one NVIDIA DGX Spark: vLLM nightly, MTP, full 262K context

A tested recipe for serving [inclusionAI/Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air) on a single DGX Spark (GB10) using the community NVFP4 build [primitive-ai/Ling-3.2-air-NVFP4](https://huggingface.co/primitive-ai/Ling-3.2-air-NVFP4), a stock vLLM nightly image (no patches, no overlay), the model's built-in MTP layer for speculative decoding, reasoning and tool calling.

Unofficial community recipe. Not affiliated with inclusionAI, Primitive, vLLM or NVIDIA. This repo has no weights.

Measured on one Spark on 2026-11-23, the day the model came out:

| | value |
|---|---|
| Script | `launch.sh` |
| Context per request | **262,144** (the model's full window; see the long-prompt note below) |
| KV pool | **1,790,211 tokens** (6.83x full-length requests at once) |
| Weights in memory | 72.79 GiB |
| MemAvailable while serving | about 14 to 19 GiB |
| Decode, one user, ~1K prompt | **36.0 to 36.5 tok/s** (MTP on) |
| Decode, one user, ~32K prompt | 30.9 to 32.3 tok/s (MTP on) |
| Prefill at 32K | about 2,500 tok/s (TTFT 13 s for a 32.5K prompt) |
| Time to first token, short prompt | 0.15 to 0.5 s |
| MTP acceptance (3 draft tokens) | 39 to 44%, mean accepted length 2.2 to 2.3 |
| Cold start to ready | 4 min 16 s |
| 69-scenario tool eval | **96.4** quality / 92.2 responsiveness / **95.1** deployability (65 pass, 3 partial, 1 fail) |
| Decode during the eval | 40.2 tok/s |

> **Unified-memory warning.** On a DGX Spark the GPU and the operating system share the same 121.6 GiB. With this launcher about 14 to 19 GiB stays available to the OS. Do not run another large model, a big build or an image/video generation job next to it. An out-of-memory on GB10 can hang the whole box until a power cycle.

> **Long prompts: measured limit so far.** Prompts up to about 33K tokens ran cleanly (TTFT 13 s). A single ~228K-token prompt pushed host MemAvailable down to 5.67 GiB during prefill. On our box two idle ComfyUI servers were also resident, and our memory guard (6 GiB floor) stopped the container. The KV pool itself has room (1.79M tokens); the pressure comes from prefill transients on top of it. Until a full-length prompt is verified on a clean box, for prompts past ~128K either keep the Spark otherwise empty, or lower `UTIL` (for example `UTIL=0.75 ./launch.sh`), or cap the window (`MAXLEN=131072 ./launch.sh`).

## 1. The model

- **Ling-3.2-air**, inclusionAI, released 2026-11-23 under the MIT license. Text, image and video in, text out. Reasoning (thinking on by default) and tool calling.
- **Size:** 124B total parameters, 5.5B active per token.
- **Layers:** 42 in total: 35 KDA linear-attention layers and 7 gated MLA layers. 512 routed experts per MoE layer (top 8) plus one shared expert. One MTP (NextN) layer for speculative decoding. A vision tower.
- **Context:** 262,144 tokens.
- **Official weights:** FP8 only, 129.3 GB, which does not fit one Spark next to a usable KV cache.
- **This build:** [primitive-ai/Ling-3.2-air-NVFP4](https://huggingface.co/primitive-ai/Ling-3.2-air-NVFP4), 77.9 GB, compressed-tensors.
  - The routed experts in layers 2 to 41 and in the MTP layer are NVFP4 (W4A4). Attention, shared experts, dense MLPs and routers are BF16 (dequantized from the FP8 release). The vision tower, embeddings and LM head are BF16 and byte-identical to the release.
  - Its card reports drift +18.85x10^-3 nats per token and 95.9% top-1 agreement against the FP8 release, measured on an RTX PRO 6000 Blackwell. This recipe is the first run of it on GB10 that we know of.

**Why the KV pool is so large for its memory:** 35 of the 42 layers are KDA linear attention, which keeps a fixed-size state per request instead of a growing cache. Only the 7 MLA layers store a (compressed) KV cache that grows with context. That is why 19 GiB of KV holds 1.79M tokens.

## 2. Hardware and software (as tested)

| Item | Value |
|---|---|
| Box | 1x NVIDIA DGX Spark, GB10 (sm_121), 121.63 GiB unified memory visible to Linux |
| OS | DGX OS 7.2.3 (OTA 7.6.0), kernel 7.0.0-1025-nvidia |
| Driver / CUDA | 580.178.04 / CUDA 13.0 |
| Image | `vllm/vllm-openai:nightly-eff4de9be3c901c6ad12cb651de64e4b10ced219`, digest `sha256:7474ce29a0708258e5de643b44bac82755dadc4c3360ffff0ff6b2d0c53e19be` (built 2026-11-23, multi-arch with arm64, 27.7 GB) |
| vLLM inside | `0.34.0rc1.dev1163+geff4de9be`, torch 2.14.0+cu130, Python 3.14.10 |
| Weights | `primitive-ai/Ling-3.2-air-NVFP4` revision `e7461aad4dfaedf0e4064d3db0e08d53ce641711` (base `inclusionAI/Ling-3.2-air` revision `1613b1a4dcc6aa0d8f346606628cd9a92061a525`) |
| Weights path | `~/models/Ling-3.2-air-NVFP4` on the host, mounted read-only at `/models/Ling-3.2-air-NVFP4` |

**About the image.** The NVFP4 model card pins `vllm/vllm-openai:nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121`. That tag was never published on Docker Hub (`docker manifest inspect` returns "no such manifest"), and a2bcb56 is a 2026-11-05 main commit. We used the 2026-11-23 nightly instead. It is 1,403 commits ahead of a2bcb56, ships an arm64 build, and includes vLLM PR #68087 "[Bugfix][Model] Fix Ling VL MTP quantization config" (2026-11-17), which matters here because the MTP layer's experts are NVFP4. The first launch worked with no errors.

## 3. Download

```bash
pip install -U "huggingface_hub[cli]"
hf download primitive-ai/Ling-3.2-air-NVFP4 \
  --revision e7461aad4dfaedf0e4064d3db0e08d53ce641711 \
  --local-dir ~/models/Ling-3.2-air-NVFP4
docker pull vllm/vllm-openai:nightly-eff4de9be3c901c6ad12cb651de64e4b10ced219
```

77.9 GB of weights (44 safetensors shards plus the model's remote code) and a 27.7 GB image. Check you have about 110 GB free.

## 4. Quick start

```bash
git clone https://github.com/tonyd2wild/Ling-3.2-air-NVFP4-DGX-Spark.git
cd Ling-3.2-air-NVFP4-DGX-Spark
./launch.sh                 # MTP on, port 8140
docker logs -f ling32air    # wait for "Application startup complete" (about 4 minutes cold)
python3 sanity.py --base http://localhost:8140
```

Options (environment variables): `PORT` (default 8140), `MODEL_DIR`, `MAXLEN` (default 262144), `UTIL` (default 0.80), `NAME` (default `ling32air`), `RESTART` (default `unless-stopped`), and `SPEC` (the speculative config; `SPEC= ./launch.sh` turns MTP off). Extra vLLM flags can be added at the end: `./launch.sh --max-num-seqs 4`.

Stop it with `./stop.sh`.

## 5. Every flag explained

| Flag | Why |
|---|---|
| `--gpus all --ipc=host` | The GPU, and shared memory for the engine's worker processes. |
| `--trust-remote-code` | The checkpoint ships its own config, tokenizer and processor code (`bailing_moe_v3_vl`). |
| `--served-model-name ling-3.2-air primitive-ai/Ling-3.2-air-NVFP4` | Either name works in requests. |
| `--max-model-len 262144` | The model's full context window. |
| `--max-num-seqs 8` | Up to 8 requests decoding at once. The KV pool can hold far more short requests; raise it if you serve many users. |
| `--gpu-memory-utilization 0.80` | Caps the instance at 97.3 GiB of the 121.6 GiB, which leaves the OS about 14 to 19 GiB. Going higher grows the KV pool but eats the OS headroom (see the warning above). |
| `--reasoning-parser ling3` | Splits the model's thinking into `reasoning_content`, so `content` holds only the answer. |
| `--enable-auto-tool-choice --tool-call-parser ling3` | OpenAI-style tool calls. |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | Uses the checkpoint's own MTP layer to draft 3 tokens per step. vLLM warns that more than one draft token reruns the same MTP layer and may lower acceptance; we measured 39 to 44% acceptance and 2.2 to 2.3 tokens accepted per step. |

Not needed: no quantization flag (vLLM reads the compressed-tensors config from the checkpoint), no KV dtype flag (BF16 KV), no patches.

## 6. Memory budget (unified memory, 121.6 GiB total)

From the startup log:

| | GiB |
|---|---|
| Instance cap (`--gpu-memory-utilization 0.80`) | 97.3 |
| Weights and non-torch memory | 72.16 (weights alone 72.79 reported at load) |
| Peak activations kept for transient use | 4.55 |
| KV cache | 18.97 (1,079 blocks) |
| CUDA graphs | 0.16 |
| Encoder cache for vision/video | 0.14 (taken from the KV cache) |
| **Left to the OS (MemAvailable while serving)** | **about 14 to 19** |

The log also says "Add 5 padding layers, may waste at most 14.29% KV cache memory". That comes from page-aligning the hybrid KDA/MLA layers, and the attention block size is forced to 2048. It is a warning, not an error.

## 7. Using it

OpenAI-compatible at `http://<spark>:8140/v1`.

Thinking is **on** by default. For fast, direct answers turn it off per request:

```bash
curl http://localhost:8140/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ling-3.2-air",
  "messages": [{"role": "user", "content": "Explain KDA linear attention in two sentences."}],
  "max_tokens": 512,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

With thinking on, the reasoning arrives in `reasoning_content` and the answer in `content`. inclusionAI recommends `temperature 1.0`, `top_p 0.95`, `top_k 20` with thinking on.

Images work through the usual `image_url` content parts.

## 8. Speed and sanity checks

Single stream, one request at a time, `bench/bench_decode.py` (stdlib; unique prompts built from public-domain novels, thinking off, temperature 0, 512 output tokens). MTP numbers come from vLLM's `/metrics` speculative-decoding counters. Raw output: [`results/spark5_decode_mtp_on.json`](results/spark5_decode_mtp_on.json).

| Prompt | TTFT | Decode | MTP acceptance | Mean accepted length |
|---|---|---|---|---|
| 1,005 tokens | 0.49 s | 36.0 tok/s | 42.0% | 2.26 |
| 1,106 tokens | 0.51 s | 36.5 tok/s | 43.9% | 2.32 |
| 32,517 tokens | 13.0 s (2,499 tok/s prefill) | 32.3 tok/s | 38.8% | 2.17 |
| 33,600 tokens | 13.3 s | 30.9 tok/s | 40.7% | 2.22 |

A plain chat check (700-token answer, thinking off, temperature 0.6, short prompt): 32.0 and 32.6 tok/s, TTFT 0.15 to 0.31 s.

The corpus is three Project Gutenberg books (pg1342, pg2701, pg98). Put the `.txt` files in `bench/corpus/` to rerun:

```bash
python3 bench/bench_decode.py --base http://localhost:8140 --model ling-3.2-air --lengths 1000 32000 --out results/decode.json
```

`sanity.py` checks a short answer with thinking off, a reasoning answer with thinking on, and one tool call. On this setup: all pass (0.27 s, 1.06 s and 0.59 s).

## 9. 69-scenario tool eval

Our 69-scenario tool-calling eval ([2Wild-Model-Eval](https://github.com/tonyd2wild/2Wild-Model-Eval); multi-turn tool loops with mock tool results, Pass/Partial/Fail = 2/1/0) against this server with `launch.sh` as is: temperature 0, max 8,192 tokens per turn, thinking left at the model's default (on), tools through `--tool-call-parser ling3`, MTP on, one request at a time. Scorecard: [`results/eval69_mtp3_262k.md`](results/eval69_mtp3_262k.md) (and `.json`).

| | NVFP4, MTP on, 262K |
|---|---|
| Quality | **96.4** / 100 (133 of 138 points) |
| Result | 65 pass, 3 partial, 1 fail |
| Responsiveness | 92.2 (median turn latency 2.03 s) |
| Deployability | **95.1** |
| Decode | 40.2 tok/s (37.5 effective) |
| Tokens | 83,816 for the whole run |

| category | pass / partial / fail |
|---|---|
| Tool Selection, Parameter Precision, Instruction Following, Safety & Boundaries, Toolset Scale (50+ tools), Multi-Step Chains, Structured Output, Contradictory Parameters, Omitted Required Parameter, Context Retention, Hallucinated Tools, Format Compliance, Refusal Calibration | all pass (60 of 60) |
| Ambiguity Handling | 2 / 2 / 0 |
| Error Recovery | 2 / 1 / 1 |

The lost points:

- **TC-34 (fail):** a tool returned an error and the model did not report it.
- **TC-33 (partial):** it reported a tool error but did not retry.
- **TC-44 and TC-67 (partial):** on an ambiguous request it made a reasonable guess instead of asking which one the user meant.

Everything else passed, including all 6 safety scenarios, all 5 refusal-calibration scenarios (no over-refusal of benign requests, clear refusals of malicious ones), and tool selection among 50+ tools.

## 10. Troubleshooting

- **"no such manifest" for the nightly in the model card:** that tag was never published. Use the image in `launch.sh`.
- **Startup looks stuck after the weights load:** torch.compile and CUDA graph capture take about 2.5 minutes on first start. Watch `docker logs -f ling32air`.
- **FlashInfer autotune messages about skipped tactics:** on GB10 (sm_121) the fused-MoE autotune skips two unsupported tactics per GEMM. Harmless.
- **The box gets slow or swaps:** something else is using memory. Stop it, or lower `UTIL` (for example `UTIL=0.75 ./launch.sh`), which shrinks the KV pool.
- **You want a smaller footprint:** lower `MAXLEN` does not save memory by itself (the pool is sized by `UTIL`). Lower `UTIL` instead.
- **The box runs out of memory on a very long prompt (200K+):** see the long-prompt note at the top. Lower `UTIL`, cap `MAXLEN`, or stop everything else on the Spark first.

## 11. Files

| File | What |
|---|---|
| `launch.sh` | Starts the server (MTP on by default). |
| `stop.sh` | Stops and removes the container. |
| `sanity.py` | Three quick checks (stdlib). |
| `bench/bench_decode.py` | Single-stream decode benchmark with MTP acceptance (stdlib). |
| `results/` | Raw benchmark output and eval scorecards. |

## Credits

- [inclusionAI](https://huggingface.co/inclusionAI) for Ling-3.2-air (MIT).
- [Primitive](https://huggingface.co/primitive-ai) for the NVFP4 build.
- The vLLM project for Ling support and the MTP quantization fix.

## License

The scripts and docs in this repo are MIT (see `LICENSE`). The model weights are not included; they are MIT under their own repos.
