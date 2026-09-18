#!/usr/bin/env python
"""
Hunyuan3D-2mv DiT LoRA 학습 속도/메모리 벤치마크.
3090 과 Colab GPU 를 같은 조건으로 비교하기 위한 스크립트.

로컬:  python bench/bench_dit.py
Colab: !pip install -q torch einops && !python bench_dit.py
       (레포 없이도 돌도록 DiT 정의를 이 파일에 포함)
"""
import argparse, time, math
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

# ---- Hunyuan3D-2mv DiT 와 동일한 블록 구조 (가중치 없이 형태만) ----
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.scale

class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.query_norm = RMSNorm(dim); self.key_norm = RMSNorm(dim)
    def forward(self, q, k, v):
        return self.query_norm(q).to(v.dtype), self.key_norm(k).to(v.dtype)

def attention(q, k, v):
    return rearrange(nn.functional.scaled_dot_product_attention(q, k, v), "B H L D -> B L (H D)")

class Modulation(nn.Module):
    def __init__(self, dim, double):
        super().__init__(); self.is_double = double; self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)
    def forward(self, vec):
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return out[:3], (out[3:] if self.is_double else None)

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super().__init__(); self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(dim // num_heads); self.proj = nn.Linear(dim, dim)

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, qkv_bias=True):
        super().__init__(); mlp_hidden = int(hidden_size * mlp_ratio); self.num_heads = num_heads
        self.img_mod = Modulation(hidden_size, True); self.txt_mod = Modulation(hidden_size, True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.img_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden, hidden_size))
        self.txt_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden, hidden_size))
    def forward(self, img, txt, vec):
        (ims, imsc, img_gate), (ims2, imsc2, img_gate2) = self.img_mod(vec)
        (tms, tmsc, txt_gate), (tms2, tmsc2, txt_gate2) = self.txt_mod(vec)
        im = (1 + imsc) * self.img_norm1(img) + ims
        iq, ik, iv = rearrange(self.img_attn.qkv(im), "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        iq, ik = self.img_attn.norm(iq, ik, iv)
        tm = (1 + tmsc) * self.txt_norm1(txt) + tms
        tq, tk, tv = rearrange(self.txt_attn.qkv(tm), "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        tq, tk = self.txt_attn.norm(tq, tk, tv)
        q = torch.cat((tq, iq), 2); k = torch.cat((tk, ik), 2); v = torch.cat((tv, iv), 2)
        attn = attention(q, k, v)
        ta, ia = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]
        img = img + img_gate * self.img_attn.proj(ia)
        img = img + img_gate2 * self.img_mlp((1 + imsc2) * self.img_norm2(img) + ims2)
        txt = txt + txt_gate * self.txt_attn.proj(ta)
        txt = txt + txt_gate2 * self.txt_mlp((1 + tmsc2) * self.txt_norm2(txt) + tms2)
        return img, txt

class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__(); self.hidden_size = hidden_size; self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh"); self.modulation = Modulation(hidden_size, False)
    def forward(self, x, vec):
        (shift, scale, gate), _ = self.modulation(vec)
        qkv, mlp = torch.split(self.linear1((1 + scale) * self.pre_norm(x) + shift),
                               [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        return x + gate * self.linear2(torch.cat((attention(q, k, v), self.mlp_act(mlp)), 2))

class DiT(nn.Module):
    """2mv config: hidden 1024, heads 16, depth 16 + single 32, in 64, context 1536"""
    def __init__(self, hidden=1024, heads=16, depth=16, depth_single=32, in_ch=64, ctx=1536):
        super().__init__()
        self.latent_in = nn.Linear(in_ch, hidden); self.cond_in = nn.Linear(ctx, hidden)
        self.time_in = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.double_blocks = nn.ModuleList([DoubleStreamBlock(hidden, heads) for _ in range(depth)])
        self.single_blocks = nn.ModuleList([SingleStreamBlock(hidden, heads) for _ in range(depth_single)])
        self.final = nn.Linear(hidden, in_ch)
    def forward(self, x, tvec, cond):
        img = self.latent_in(x); txt = self.cond_in(cond); vec = self.time_in(tvec)
        for b in self.double_blocks: img, txt = b(img, txt, vec)
        h = torch.cat((txt, img), 1)
        for b in self.single_blocks: h = b(h, vec)
        return self.final(h[:, txt.shape[1]:])

class LoRALinear(nn.Module):
    def __init__(self, base, r=8):
        super().__init__(); self.base = base
        dt, dv = base.weight.dtype, base.weight.device
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dv, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dv, dtype=dt))
        nn.init.normal_(self.A, std=0.01)
    def forward(self, x): return self.base(x) + (x @ self.A.T) @ self.B.T

def wrap_ckpt(mod_list):
    class W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, *a): return checkpoint(self.m, *a, use_reentrant=False)
    return nn.ModuleList([W(m) for m in mod_list])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--views', type=int, default=4)
    ap.add_argument('--patches', type=int, default=1369, help='DINOv2-giant 518px = 37x37')
    ap.add_argument('--latents', type=int, default=3072)
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--no_ckpt', action='store_true', help='gradient checkpointing 끄기')
    ap.add_argument('--iters', type=int, default=3, help='측정 반복 (워밍업 1회 제외)')
    a = ap.parse_args()

    assert torch.cuda.is_available(), 'GPU 없음'
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    print(f'GPU: {name} | compute {cap[0]}.{cap[1]} | {total:.0f} GiB | dtype {str(dtype).split(".")[-1]}')
    print(f'조건: latent {a.latents} tok, cond {a.views}x{a.patches}={a.views*a.patches} tok, '
          f'LoRA rank {a.rank}, checkpointing {"off" if a.no_ckpt else "on"}')

    model = DiT().to('cuda', dtype)
    for p in model.parameters(): p.requires_grad_(False)
    n = 0
    for blk in model.double_blocks:
        for m in (blk.img_attn, blk.txt_attn):
            for nm in ('qkv', 'proj'): setattr(m, nm, LoRALinear(getattr(m, nm), a.rank)); n += 1
    for blk in model.single_blocks:
        for nm in ('linear1', 'linear2'): setattr(blk, nm, LoRALinear(getattr(blk, nm), a.rank)); n += 1
    if not a.no_ckpt:
        model.double_blocks = wrap_ckpt(model.double_blocks)
        model.single_blocks = wrap_ckpt(model.single_blocks)
    tr = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(tr, lr=1e-4)
    print(f'LoRA Linear {n}개, 학습 파라미터 {sum(p.numel() for p in tr)/1e6:.1f}M\n')

    print(f'{"batch":>6} {"peak GiB":>10} {"s/step":>8} {"s/sample":>9} {"samples/h":>10}')
    for B in a.batches:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(B, a.latents, 64, device='cuda', dtype=dtype)
            tv = torch.randn(B, 256, device='cuda', dtype=dtype)
            cond = torch.randn(B, a.views * a.patches, 1536, device='cuda', dtype=dtype)
            for i in range(a.iters + 1):
                if i == 1: torch.cuda.synchronize(); t0 = time.time()
                loss = (model(x, tv, cond).float() - 1.0).pow(2).mean()
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / a.iters
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f'{B:>6} {peak:>10.1f} {dt:>8.2f} {dt/B:>9.2f} {3600/(dt/B):>10.0f}')
            del x, tv, cond, loss
        except torch.cuda.OutOfMemoryError:
            print(f'{B:>6} {"OOM":>10}'); torch.cuda.empty_cache(); break

if __name__ == '__main__':
    main()
