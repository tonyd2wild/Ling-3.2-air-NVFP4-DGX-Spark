# Quick timing of the MLA decode kernel (8 seqs x 4 rows = MTP verify shape).
import os, torch, triton
from vllm.v1.attention.ops import triton_decode_attention as tda
dev="cuda"; H, LK, LV, PAGE = 32, 576, 512, 3968
props = torch.cuda.get_device_properties(0)
def bench(kv_dtype, n, R=32):
    pages_per = (n + PAGE - 1)//PAGE; nreq = R//4; npages = nreq*pages_per+1
    kv = torch.empty(npages, PAGE, 1, LK, device=dev, dtype=torch.bfloat16).normal_(0, 0.5).to(kv_dtype)
    q = (torch.randn(R, H, LK, device=dev)*0.5).to(torch.bfloat16)
    bt = ((torch.arange(nreq*pages_per, device=dev)+1).view(nreq, pages_per).repeat_interleave(4, 0)).to(torch.int32)
    sl = torch.full((R,), n, dtype=torch.int32, device=dev)
    sp = min(triton.next_power_of_2(max(1, n//512)), props.multi_processor_count*2)
    o = torch.zeros(R, H, LV, dtype=torch.bfloat16, device=dev); lse = torch.zeros(R, H, dtype=torch.bfloat16, device=dev)
    lg = torch.empty(R, H, sp, LV+1, dtype=torch.float32, device=dev); ks = torch.tensor(1.0, device=dev)
    f = lambda: tda.decode_attention_fwd(q, kv, kv[..., :LV], o, lse, bt, sl, lg, sp, 0.04, PAGE, k_scale=ks, v_scale=ks, is_mla=True)
    for _ in range(3): f()
    torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True); s.record()
    for _ in range(10): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/10
tag = os.environ.get("TAG","")
for n in (1024, 32768, 131072):
    print(f"{tag:22s} ctx={n:6d} bf16={bench(torch.bfloat16,n):7.3f} ms  fp8={bench(torch.float8_e4m3fn,n):7.3f} ms")
