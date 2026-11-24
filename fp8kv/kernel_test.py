# Standalone check of the Triton MLA grouped decode kernel with an FP8 KV cache
# (Ling-3.2-air MLA shapes: 32 heads, kv_lora_rank 512, rope 64 -> Lk 576).
import os, sys, time, torch
from vllm.v1.attention.ops import triton_decode_attention as tda

torch.manual_seed(0)
dev = "cuda"
H, LK, LV = 32, 576, 512
PAGE = int(os.environ.get("PAGE", "3968"))
fp8 = torch.float8_e4m3fn
props = torch.cuda.get_device_properties(0)
print("device", props.name, "cap", props.major, props.minor,
      "smem_optin", getattr(props, "shared_memory_per_block_optin", None),
      "sms", props.multi_processor_count)

def run(seq_lens, kv_dtype, splits):
    R = len(seq_lens)
    max_len = max(seq_lens)
    pages_per = (max_len + PAGE - 1) // PAGE
    npages = R * pages_per + 1
    kv_bf = (torch.randn(npages, PAGE, 1, LK, device=dev) * 0.5).to(torch.bfloat16)
    if kv_dtype == fp8:
        kv = kv_bf.to(fp8)
        kv_ref = kv.float()
    else:
        kv = kv_bf
        kv_ref = kv_bf.float()
    q = (torch.randn(R, H, LK, device=dev) * 0.5).to(torch.bfloat16)
    perm = torch.randperm(npages - 1, device=dev) + 1
    bt = perm[: R * pages_per].view(R, pages_per).to(torch.int32)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    o = torch.zeros(R, H, LV, dtype=torch.bfloat16, device=dev)
    lse = torch.zeros(R, H, dtype=torch.bfloat16, device=dev)
    logits = torch.empty(R, H, splits, LV + 1, dtype=torch.float32, device=dev)
    scale = 1.0 / (LK ** 0.5)
    ks = torch.tensor(1.0, dtype=torch.float32, device=dev)
    tda.decode_attention_fwd(q, kv, kv[..., :LV], o, lse, bt, sl, logits, splits,
                             scale, PAGE, k_scale=ks, v_scale=ks, is_mla=True)
    torch.cuda.synchronize()
    err = 0.0
    for r in range(R):
        n = seq_lens[r]
        tok = torch.arange(n, device=dev)
        pg = bt[r, tok // PAGE].long()
        k = kv_ref[pg, tok % PAGE, 0, :]          # [n, 576]
        s = (q[r].float() @ k.T) * scale           # [H, n]
        p = torch.softmax(s, -1)
        ref = p @ k[:, :LV]
        err = max(err, (o[r].float() - ref).abs().max().item())
    return err

def splits_for(n):
    import triton
    return min(triton.next_power_of_2(max(1, n // 512)), props.multi_processor_count * 2)

cases = [[1], [37, 4000], [8191, 300, 16384, 1], [70000] * 2]
for name, dt in (("bf16", torch.bfloat16), ("fp8_e4m3", fp8)):
    for sls in cases:
        sp = splits_for(max(sls))
        try:
            e = run(sls, dt, sp)
            print(f"{name:9s} seqs={sls} splits={sp} max_abs_err_vs_ref={e:.4f} OK")
        except Exception as ex:
            print(f"{name:9s} seqs={sls} splits={sp} FAILED {type(ex).__name__}: {str(ex)[:160]}")
