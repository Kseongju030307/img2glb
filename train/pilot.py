#!/usr/bin/env python
"""파이프라인 전체를 소규모로 끝까지 돌려 loss 가 내려가는지 확인하는 스모크 학습."""
import os, sys, json, math, time, gc
from pathlib import Path
import numpy as np, torch, torch.nn as nn, trimesh
from PIL import Image
from torch.utils.checkpoint import checkpoint
from scipy.spatial import cKDTree

REPO = Path('/data/sungju030307/img2glb/Hunyuan3D-2'); sys.path.insert(0, str(REPO))
os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
DATA = Path('/data/sungju030307/img2glb/train'); (DATA/'cache').mkdir(parents=True, exist_ok=True)
DEV = 'cuda:0'
N_MODELS, STEPS, BATCH, RANK, ALPHA = int(sys.argv[1]), int(sys.argv[2]), 2, 8, 16
VIEW_AZIM = {'front':0, 'left':90, 'back':180, 'right':270}
CHAMFER_MAX, MIN_FACES = 0.025, 20000

sys.path.insert(0, '/home/sungju030307/img2glb')
from bench.render_views import render_ortho   # 검증된 직교 렌더러

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.models.autoencoders import ShapeVAE
from hy3dgen.shapegen.surface_loaders import SharpEdgeSurfaceLoader

print('== 모델 로드', flush=True)
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2mv',
        subfolder='hunyuan3d-dit-v2-mv', variant='fp16', device=DEV)
dit, conditioner, sched = pipe.model, pipe.conditioner, pipe.scheduler
DT = next(dit.parameters()).dtype
for p in dit.parameters(): p.requires_grad_(False)
for p in conditioner.parameters(): p.requires_grad_(False)
vae = ShapeVAE.from_pretrained('tencent/Hunyuan3D-2', subfolder='hunyuan3d-vae-v2-0-withencoder',
        use_safetensors=False, pc_size=30720, pc_sharpedge_size=30720).to(DEV).eval()
loader = SharpEdgeSurfaceLoader(num_sharp_points=30720, num_uniform_points=30720)
print(f'   DiT {sum(p.numel() for p in dit.parameters())/1e9:.2f}B dtype {DT}', flush=True)

# ---------- 전처리 ----------
import objaverse
recs = json.load(open('/data/sungju030307/objaverse_meta/hiprec_anime.json'))
def pts(m, n=30000):
    v,_ = trimesh.sample.sample_surface(m, n); v = np.asarray(v)-m.bounds.mean(0)
    return v/np.abs(v).max()

@torch.no_grad()
def cond_of(imgs):
    ci = pipe.prepare_image(imgs)
    img = ci.pop('image').to(DEV, dtype=DT)
    return pipe.encode_cond(image=img, additional_cond_inputs=ci,
                            do_classifier_free_guidance=False, dual_guidance=False)['main']

print('== 전처리', flush=True)
ok, tried, t0 = 0, 0, time.time()
for r in recs:
    if ok >= N_MODELS: break
    tried += 1
    d = DATA/'cache'/r['uid']
    if (d/'latent.npy').exists() and (d/'cond.npy').exists(): ok += 1; continue
    try:
        p = objaverse.load_objects(uids=[r['uid']], download_processes=1)[r['uid']]
        sc = trimesh.load(p, process=False)
        m = sc.to_geometry() if hasattr(sc,'to_geometry') else sc
        del sc; gc.collect()
        if m.is_empty or len(m.faces) < MIN_FACES: continue
        if len(m.faces) > 200000: m = m.simplify_quadric_decimation(face_count=200000)
        m.apply_translation(-m.bounds.mean(0)); m.apply_scale(1.0/np.abs(m.vertices).max())
        d.mkdir(parents=True, exist_ok=True); g = d/'mesh.glb'; m.export(g)
        with torch.no_grad():
            lat = vae.encode(loader(str(g)).to(DEV, dtype=torch.float16))
            o = vae.latents2mesh(vae.decode(lat), output_type='trimesh', bounds=1.01, mc_level=0.0,
                                 num_chunks=20000, octree_resolution=256, mc_algo='mc', enable_pbar=False)[0]
        rt = trimesh.Trimesh(vertices=o.mesh_v, faces=o.mesh_f)
        if rt.is_empty or len(rt.faces) < 100: raise RuntimeError('빈 복원')
        A,B = pts(m), pts(rt)
        err = float((cKDTree(B).query(A)[0].mean()+cKDTree(A).query(B)[0].mean())/2)
        if err > CHAMFER_MAX:
            print(f'   거름 {r["name"][:22]:22s} chamfer {err:.4f}', flush=True); continue
        rr = m.simplify_quadric_decimation(face_count=120000) if len(m.faces)>120000 else m
        imgs = {k: render_ortho(rr, a, 512) for k,a in VIEW_AZIM.items()}
        np.save(d/'latent.npy', lat[0].float().cpu().numpy().astype(np.float16))
        np.save(d/'cond.npy', cond_of(imgs)[0].float().cpu().numpy().astype(np.float16))
        ok += 1
        print(f'   [{ok}/{N_MODELS}] {r["name"][:22]:22s} faces {len(m.faces):>6} chamfer {err:.4f}', flush=True)
    except Exception as e:
        print(f'   실패 {r["name"][:22]:22s} {type(e).__name__}', flush=True)
print(f'   전처리 완료 {ok}개 / 시도 {tried}개 / {time.time()-t0:.0f}s', flush=True)
del vae; gc.collect(); torch.cuda.empty_cache()

# ---------- LoRA ----------
class LoRALinear(nn.Module):
    """LoRA 가중치는 fp32 로 유지한다.
    base 가 fp16 이라고 A/B 도 fp16 으로 두면 AdamW 의 eps(1e-8)가 fp16 에서 0 으로
    언더플로해서 1스텝 만에 NaN 이 된다. (실측 확인)"""
    def __init__(self, base, r, alpha, attn_out=None):
        super().__init__(); self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        dv = base.weight.device
        self.out_slice = attn_out
        of = attn_out or base.out_features
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dv, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(of, r, device=dv, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); self.scale = alpha/r
    def forward(self, x):
        y = self.base(x)
        d = ((x.float() @ self.A.T) @ self.B.T * self.scale).to(y.dtype)
        if self.out_slice:
            y = y.clone(); y[..., :self.out_slice] = y[..., :self.out_slice] + d; return y
        return y + d
hid = dit.hidden_size; n = 0
for blk in dit.double_blocks:
    for at in (blk.img_attn, blk.txt_attn):
        for nm in ('qkv','proj'): setattr(at, nm, LoRALinear(getattr(at,nm), RANK, ALPHA)); n += 1
for blk in dit.single_blocks:
    blk.linear1 = LoRALinear(blk.linear1, RANK, ALPHA, attn_out=3*hid); n += 1
    blk.linear2 = LoRALinear(blk.linear2, RANK, ALPHA); n += 1
lora = [p for p in dit.parameters() if p.requires_grad]
print(f'== LoRA {n}개 Linear, {sum(p.numel() for p in lora)/1e6:.2f}M 파라미터', flush=True)
class Ck(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, *a, **kw):
        ks = list(kw); vs = list(a)+[kw[k] for k in ks]; na = len(a)
        return checkpoint(lambda *v: s.m(*v[:na], **dict(zip(ks, v[na:]))), *vs, use_reentrant=False)
dit.double_blocks = nn.ModuleList([Ck(b) for b in dit.double_blocks])
dit.single_blocks = nn.ModuleList([Ck(b) for b in dit.single_blocks])

# ---------- 학습 ----------
items = [d for d in sorted((DATA/'cache').iterdir()) if (d/'latent.npy').exists() and (d/'cond.npy').exists()]
print(f'== 학습 시작: 샘플 {len(items)}개, {STEPS} 스텝, batch {BATCH}', flush=True)
opt = torch.optim.AdamW(lora, lr=1e-4, weight_decay=0.01, betas=(0.9,0.99))
UNCOND = None; hist = []; t0 = time.time(); rng = np.random.RandomState(0)
for step in range(1, STEPS+1):
    idx = rng.choice(len(items), BATCH, replace=len(items) < BATCH)
    x0 = torch.stack([torch.from_numpy(np.load(items[i]/'latent.npy')) for i in idx]).to(DEV, DT)
    cond = torch.stack([torch.from_numpy(np.load(items[i]/'cond.npy')) for i in idx]).to(DEV, DT)
    sig32 = torch.rand(BATCH, device=DEV, dtype=torch.float32)          # 타임스텝은 float32
    s = sig32.to(DT).view(BATCH,1,1)
    noise = torch.randn_like(x0); x_t = s*noise + (1-s)*x0; target = noise - x0
    m = torch.rand(BATCH, device=DEV) < 0.1
    if m.any():
        if UNCOND is None:
            with torch.no_grad():
                UNCOND = conditioner.unconditional_embedding(BATCH, view_idxs=[(0,1,2,3)]*BATCH)['main'].to(DEV, DT)
        cond = cond.clone(); cond[m] = UNCOND[m]
    loss = torch.nn.functional.mse_loss(dit(x_t, sig32*1000, {'main': cond}).float(), target.float())
    loss.backward(); torch.nn.utils.clip_grad_norm_(lora, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True); hist.append(loss.item())
    if step % 10 == 0:
        print(f'   step {step:>4}/{STEPS}  loss {np.mean(hist[-10:]):.4f}  '
              f'{(time.time()-t0)/step:.2f}s/step  peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB', flush=True)
h = np.array(hist); k = max(len(h)//5, 5)
print(f'\n== 결과: 처음 {k}스텝 평균 {h[:k].mean():.4f} -> 마지막 {k}스텝 평균 {h[-k:].mean():.4f} '
      f'({(1-h[-k:].mean()/h[:k].mean())*100:+.1f}%)', flush=True)
torch.save({'lora':{k2:v.detach().cpu() for k2,v in dit.state_dict().items() if ('.A' in k2 or '.B' in k2)},
            'hist':hist}, DATA/'pilot_lora.pt')
np.save(DATA/'pilot_hist.npy', h)
print('저장:', DATA/'pilot_lora.pt', flush=True)
