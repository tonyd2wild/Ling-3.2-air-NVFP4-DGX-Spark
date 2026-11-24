# Ling-3.2-air (NVFP4) on one NVIDIA DGX Spark: vLLM nightly, MTP, FP8 KV cache, full 262K context

A tested recipe for serving [inclusionAI/Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air) on a single DGX Spark (GB10) using the community NVFP4 build [primitive-ai/Ling-3.2-air-NVFP4](https://huggingface.co/primitive-ai/Ling-3.2-air-NVFP4), a vLLM nightly image, the model's built-in MTP layer for speculative decoding, reasoning and tool calling.

Two launchers:

- **`launch_fp8.sh` (recommended, added 2026-11-24):** FP8 KV cache. Needs two small vLLM patches, built once with `./fp8kv/prepare.sh` and mounted read-only over the image copies (section 6).
- **`launch.sh`:** the stock image as is, BF16 KV cache, no patches.

Unofficial community recipe. Not affiliated with inclusionAI, Primitive, vLLM or NVIDIA. This repo has no weights.

Measured on one Spark on 2026-11-23 and 2026-11-24:

| | **FP8 KV (`launch_fp8.sh`)** | BF16 KV (`launch.sh`) |
|---|---|---|
| Patches | 2, mounted read-only (section 6) | none |
| `--gpu-memory-utilization` | 0.765 | 0.80 |
| Context per request | **262,144** | 262,144 |
| KV pool | **2,170,119 tokens** (8.28x full-length requests at once) | 1,790,211 tokens (6.83x) |
| KV cache memory | 13.67 GiB (about 6.6 KiB per token) | 18.97 GiB (about 11.1 KiB per token) |
| MemAvailable while serving, idle | **about 22 GiB** | about 14 to 19 GiB |
| One ~215K-token prompt | **answered correctly in 158 s; lowest MemAvailable 14.5 GiB** | a ~228K prompt dropped MemAvailable to 5.67 GiB and our 6 GiB memory guard stopped it |
| Decode, one user, ~1K prompt | **35.8 tok/s** (MTP on) | **36.0 to 36.5 tok/s** (MTP on) |
| MTP acceptance (3 draft tokens) | 43.7%, mean accepted length 2.31 | 42.0 to 43.9%, 2.26 to 2.32 |
| Decode, one user, ~32K prompt | not re-measured | 30.9 to 32.3 tok/s |
| Prefill at 32K | not re-measured | about 2,500 tok/s (TTFT 13 s) |
| Checks | 10/10 sanity set, thinking-off math 2/2, vision 1/1 | `sanity.py` 3/3 |
| 69-scenario tool eval | not re-run | **96.4** quality / 92.2 responsiveness / **95.1** deployability |

Both: weights 72.79 GiB in memory, time to first token 0.15 to 0.5 s on short prompts, cold start to ready about 4 minutes.

> **Unified-memory warning.** On a DGX Spark the GPU and the operating system share the same 121.6 GiB. Do not run another large model, a big build or an image/video generation job next to this server. An out-of-memory on GB10 can hang the whole box until a power cycle.

> **Long prompts.** With `launch_fp8.sh` a single 215,354-token prompt (880K characters of three novels) finished in 158.2 s and returned the right answer; the lowest MemAvailable during prefill was 14,844 MiB. What made the difference is the torch allocator cap (section 6), not only the smaller KV. Not tested yet: a completely full pool (all 2.17M tokens, about 13.7 GiB committed) while a long prompt is being prefilled. That would take MemAvailable much lower, so if you plan several long conversations at once, keep the Spark otherwise empty or use `UTIL=0.74 ./launch_fp8.sh`. With `launch.sh` (no patches), prompts past ~128K can still exhaust host memory; lower `UTIL` or cap the window (`MAXLEN=131072 ./launch.sh`).

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
./fp8kv/prepare.sh          # once: builds the two patched vLLM files from the pinned image
./launch_fp8.sh             # FP8 KV, MTP on, port 8140   (or ./launch.sh for the stock BF16 KV setup)
docker logs -f ling32air    # wait for "Application startup complete" (about 4 minutes cold)
python3 sanity.py --base http://localhost:8140
```

`prepare.sh` copies the two stock files out of the image (`docker create` + `docker cp`, nothing runs), checks their sha256 against the image we tested, applies the patches in `fp8kv/`, and checks the result. It refuses to continue if the image's files differ.

Options (environment variables): `PORT` (default 8140), `MODEL_DIR`, `MAXLEN` (default 262144), `UTIL` (default 0.765 for `launch_fp8.sh`, 0.80 for `launch.sh`), `NAME` (default `ling32air`), `RESTART` (default `unless-stopped`), `SPEC` (the speculative config; `SPEC= ./launch_fp8.sh` turns MTP off) and, for `launch_fp8.sh` only, `TORCH_GROWTH_GIB` (default 10; empty turns the cap off). Extra vLLM flags can be added at the end: `./launch_fp8.sh --max-num-seqs 4`.

Stop it with `./stop.sh`.

## 5. Every flag explained

| Flag | Why |
|---|---|
| `--gpus all --ipc=host` | The GPU, and shared memory for the engine's worker processes. |
| `--trust-remote-code` | The checkpoint ships its own config, tokenizer and processor code (`bailing_moe_v3_vl`). |
| `--served-model-name ling-3.2-air primitive-ai/Ling-3.2-air-NVFP4` | Either name works in requests. |
| `--max-model-len 262144` | The model's full context window. |
| `--max-num-seqs 8` | Up to 8 requests decoding at once. The KV pool can hold far more short requests; raise it if you serve many users. |
| `--gpu-memory-utilization 0.765` (FP8) / `0.80` (BF16) | Caps the instance at 93.04 / 97.3 GiB of the 121.6 GiB. Going higher grows the KV pool but eats the OS headroom (see the warning above). With FP8 KV we spent part of the saving on a bigger pool and part on more OS headroom. |
| `--kv-cache-dtype fp8` (`launch_fp8.sh`) | Stores the MLA KV cache in FP8: about 6.6 KiB per token instead of 11.1. Needs the kernel patch in section 6 on GB10. |
| `-e FP8FIX_TORCH_GROWTH_GIB=10` (`launch_fp8.sh`) | Read by the patched `extensible_kv_cache.py` (section 6). Without the variable that file behaves like stock vLLM. |
| `--reasoning-parser ling3` | Splits the model's thinking into `reasoning_content`, so `content` holds only the answer. |
| `--enable-auto-tool-choice --tool-call-parser ling3` | OpenAI-style tool calls. |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | Uses the checkpoint's own MTP layer to draft 3 tokens per step. vLLM warns that more than one draft token reruns the same MTP layer and may lower acceptance; we measured 39 to 44% acceptance and 2.2 to 2.3 tokens accepted per step. |

Not needed: no quantization flag (vLLM reads the compressed-tensors config from the checkpoint). `launch.sh` needs no patches and no KV dtype flag (BF16 KV).

## 6. FP8 KV cache: the two patches, and the memory budget

**Why patches.** On this image, `--kv-cache-dtype fp8` alone crashes at engine start on GB10 with Triton `OutOfResources: out of resource: shared memory, Required: 102400, Hardware limit: 101376`. Fixing that let the server start, but our first long-prompt test then ran the host out of memory for a second reason. One patch per problem, both in `fp8kv/` as plain diffs against the pinned image:

1. **`triton_decode_attention.patch`: MLA decode kernel shared memory.** For an FP8 KV cache, `_decode_grouped_att_m_fwd` launches the grouped MLA decode kernel with `num_stages=2`. The FP8 tile plus its BF16 dequant copies then need 102,400 B of shared memory per block, 1 KB more than GB10's opt-in limit of 101,376 B. The patch uses `num_stages=1` when the KV cache is FP8 and the GPU's opt-in shared memory is under 110 KB (GB10 sm_121, and per the same check sm_120, sm_89 and sm_86). `fp8kv/kernel_test.py` checks the kernel against an FP32 reference with Ling's MLA shapes (32 heads, Lk 576); `fp8kv/kernel_time.py` times variants: one stage with BLOCK_N 32 was the fastest that fits (two stages with BLOCK_N 16 was about 30% slower, BLOCK_N 64 does not fit). Both run inside the container with the patched file mounted, for example `docker exec -i ling32air python3 - < fp8kv/kernel_test.py`. Two env knobs exist for testing only: `FP8FIX_MLA_NUM_STAGES`, `FP8FIX_MLA_BLOCK_N`.
2. **`extensible_kv_cache.patch`: cap on torch allocator growth.** After warmup, stock vLLM lets the torch caching allocator grow into all memory that is still free after the KV cache (minus 0.5 GiB). That limit is not tied to `--gpu-memory-utilization`: here it was 99.8 GiB while the instance cap was 93.04 GiB. On unified memory, a 215K-token prefill grew and fragmented the allocator until host MemAvailable fell to 6 GiB and our memory guard stopped the container. (The earlier 228K failure with BF16 KV looks like the same mechanism; we have not re-tested BF16 with the cap.) The patch reads `FP8FIX_TORCH_GROWTH_GIB` and caps the reservation at the post-warmup reservation plus that many GiB (83.96 GiB here). At the cap torch frees cached blocks and retries instead of growing. If the variable is not set, nothing changes.

Look for these lines in `docker logs ling32air`:

```
[fp8fix] capping torch reservation at 83.96 GiB (was 99.8 GiB): post-warmup reservation + FP8FIX_TORCH_GROWTH_GIB=10
Memory after warmup: 40.0 GiB free, 0.03 GiB committed to the KV cache, 4.55 GiB kept for transient activations ...; 803 KV cache blocks (13.67 GiB) fit within the requested 93.04 GiB. Torch may reserve up to 83.96 GiB.
GPU KV cache size: 2,170,119 tokens, Maximum concurrency for 262,144 tokens per request: 8.28x
```

Note "0.03 GiB committed": this vLLM build backs the KV cache with memory that is committed as it fills, so an idle server leaves more free memory than the pool size suggests. Plan for the full 13.67 GiB when the pool is busy.

**Memory budget** (unified memory, 121.6 GiB total), from the startup logs:

| | FP8 KV, UTIL 0.765 | BF16 KV, UTIL 0.80 |
|---|---|---|
| Instance cap | 93.04 GiB | 97.3 GiB |
| Weights and non-torch memory | 73.27 GiB | 72.16 GiB (weights alone: 72.79 GiB at load in both) |
| Peak activations kept for transient use | 4.55 | 4.55 |
| KV cache | 13.67 GiB (803 blocks), 2,170,119 tokens | 18.97 GiB (1,079 blocks), 1,790,211 tokens |
| CUDA graphs | 0.13 | 0.16 |
| Torch allocator may grow to | 83.96 GiB (capped) | stock limit |
| **MemAvailable while serving, idle** | **about 22 GiB** | **about 14 to 19 GiB** |

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

**FP8 KV (`launch_fp8.sh`), 2026-11-24:**

| Check | Result |
|---|---|
| Decode, 1,088-token prompt, 512 out, thinking off | TTFT 0.52 s (2,081 tok/s prefill), **35.8 tok/s**, MTP acceptance 43.7%, mean accepted length 2.31 ([`results/spark5_fp8kv_decode_1k.json`](results/spark5_fp8kv_decode_1k.json)) |
| One 215,354-token prompt (`bench/longprompt.py 880000`) | correct answer, 158.2 s, lowest MemAvailable 14,844 MiB, memory brake not triggered ([`results/spark5_fp8kv_longprompt_215k.json`](results/spark5_fp8kv_longprompt_215k.json)) |
| 10-prompt sanity set (2 math, 2 code with unit tests, parallel tool calls, long-document summary, 2 chat, 2 vision), thinking on | 10/10 pass ([`results/spark5_fp8kv_sanity10.md`](results/spark5_fp8kv_sanity10.md)) |
| Thinking-off math x2, one-image vision question | all pass |

Decode speed is the same as BF16 KV within noise. The 32K decode and the 69-scenario eval have not been re-run with FP8 KV yet.

## 9. 69-scenario tool eval

Our 69-scenario tool-calling eval ([2Wild-Model-Eval](https://github.com/tonyd2wild/2Wild-Model-Eval); multi-turn tool loops with mock tool results, Pass/Partial/Fail = 2/1/0) against this server with `launch.sh` as is (BF16 KV): temperature 0, max 8,192 tokens per turn, thinking left at the model's default (on), tools through `--tool-call-parser ling3`, MTP on, one request at a time. Scorecard: [`results/eval69_mtp3_262k.md`](results/eval69_mtp3_262k.md) (and `.json`).

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
- **`OutOfResources: out of resource: shared memory, Required: 102400, Hardware limit: 101376` with `--kv-cache-dtype fp8`:** the stock MLA decode kernel on GB10. Use `./fp8kv/prepare.sh` + `./launch_fp8.sh` (section 6).
- **`prepare.sh` says the stock file is not the version the patches were made for:** you are on a different vLLM image. Use the pinned one, or read the two small diffs in `fp8kv/` and apply them by hand.
- **Startup looks stuck after the weights load:** torch.compile and CUDA graph capture take about 2.5 minutes on first start. Watch `docker logs -f ling32air`.
- **FlashInfer autotune messages about skipped tactics:** on GB10 (sm_121) the fused-MoE autotune skips two unsupported tactics per GEMM. Harmless.
- **The box gets slow or swaps:** something else is using memory. Stop it, or lower `UTIL` (for example `UTIL=0.75 ./launch.sh`), which shrinks the KV pool.
- **You want a smaller footprint:** lower `MAXLEN` does not save memory by itself (the pool is sized by `UTIL`). Lower `UTIL` instead.
- **The box runs out of memory on a very long prompt (200K+):** use `launch_fp8.sh` (it caps the torch allocator). With `launch.sh`, lower `UTIL`, cap `MAXLEN`, or stop everything else on the Spark first.

## 11. Files

| File | What |
|---|---|
| `launch_fp8.sh` | Starts the server with FP8 KV and the two patched files (MTP on by default). |
| `fp8kv/prepare.sh` | Builds `fp8kv/patched/` from the pinned image and the two diffs, with sha256 checks. |
| `fp8kv/*.patch` | The two vLLM diffs (section 6). |
| `fp8kv/kernel_test.py`, `fp8kv/kernel_time.py` | Correctness check against an FP32 reference, and timing, for the patched MLA decode kernel. Run inside the container. |
| `launch.sh` | Starts the stock server, BF16 KV, no patches (MTP on by default). |
| `stop.sh` | Stops and removes the container. |
| `sanity.py` | Three quick checks (stdlib). |
| `bench/bench_decode.py` | Single-stream decode benchmark with MTP acceptance (stdlib). |
| `bench/longprompt.py` | One very long prompt with a MemAvailable brake; run it on the Spark (stdlib). |
| `results/` | Raw benchmark output and eval scorecards. |

## Credits

- [inclusionAI](https://huggingface.co/inclusionAI) for Ling-3.2-air (MIT).
- [Primitive](https://huggingface.co/primitive-ai) for the NVFP4 build.
- The vLLM project for Ling support and the MTP quantization fix.

## License

The scripts and docs in this repo are MIT (see `LICENSE`). The two files in `fp8kv/*.patch` are changes to vLLM source code, which is licensed under the Apache License 2.0; those diffs are offered under the same Apache License 2.0. `prepare.sh` builds the patched files from your own copy of the vLLM image, and this repo does not include vLLM source files. The model weights are not included; they are MIT under their own repos.
