#!/usr/bin/env python
"""
학습된 LoRA 와 base 를 같은 4뷰 입력으로 생성해 비교한다.

  python train/evaluate.py --input inputs/sample1_character --gpu 0
"""
import argparse, os, sys, json, math, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--input',  default='/home/sungju030307/img2glb/inputs/sample1_character')
ap.add_argument('--ckpt',   default='/data/sungju030307/img2glb/train/ckpt/lora_last.pt')
ap.add_argument('--out',    default='/home/sungju030307/img2glb/generated')
ap.add_argument('--gpu',    type=int, default=0)
ap.add_argument('--steps',  type=int, default=50)
ap.add_argument('--octree', type=int, default=380)
ap.add_argument('--seed',   type=int, default=1234)
ap.add_argument('--scales', default='0,1', help='비교할 LoRA 세기 (쉼표)')
a = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
REPO = Path('/data/sungju030307/img2glb/Hunyuan3D-2')
sys.path.insert(0, str(REPO)); sys.path.insert(0, '/home/sungju030307/img2glb')

import numpy as np, torch, torch.nn as nn, trimesh, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree
from PIL import Image
from bgremove import remove_background
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

DEV = 'cuda:0'
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
def log(*m): print(time.strftime('[%H:%M:%S]'), *m, flush=True)

log('모델 로드')
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv', subfolder='hunyuan3d-dit-v2-mv', variant='fp16', device=DEV)
dit = pipe.model
for p in dit.parameters(): p.requires_grad_(False)

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, attn_out=None):
        super().__init__(); self.base = base
        dv = base.weight.device; self.out_slice = attn_out
        of = attn_out or base.out_features
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dv, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(of, r, device=dv, dtype=torch.float32))
        self.r, self.alpha, self.scale = r, alpha, alpha/r
    def forward(self, x):
        y = self.base(x)
        A = self.A.to(x.dtype); B = self.B.to(x.dtype)
        d = (x @ A.T) @ B.T * self.scale
        if self.out_slice:
            y = y.clone(); y[..., :self.out_slice] = y[..., :self.out_slice] + d; return y
        return y + d

ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
cargs = ck.get('args', {}); rank = cargs.get('rank', 8); alpha = cargs.get('alpha', 16)
targets = cargs.get('targets', 'attn')
log(f'체크포인트: step {ck["step"]}, loss {ck["loss"]:.4f}, rank {rank}, alpha {alpha}, targets {targets}')

hid = dit.hidden_size
for blk in dit.double_blocks:
    for at in (blk.img_attn, blk.txt_attn):
        for nm in ('qkv','proj'): setattr(at, nm, LoRALinear(getattr(at,nm), rank, alpha))
    if targets == 'attn+mlp':
        for mlp in (blk.img_mlp, blk.txt_mlp):
            for i in (0,2): mlp[i] = LoRALinear(mlp[i], rank, alpha)
for blk in dit.single_blocks:
    blk.linear1 = LoRALinear(blk.linear1, rank, alpha, attn_out=(3*hid if targets=='attn' else None))
    blk.linear2 = LoRALinear(blk.linear2, rank, alpha)
# 학습 때는 gradient checkpointing 래퍼(CkptWrap) 안에 있어서 키에 ".m." 이 끼어 있다.
# 평가에서는 래퍼를 안 쓰므로 그 조각을 떼어내고 맞춰야 한다. 안 그러면
# strict=False 때문에 아무것도 안 실린 채 조용히 넘어간다.
sd = {k.replace('.m.', '.', 1) if '.m.' in k else k: v for k, v in ck['lora'].items()}
own = set(dit.state_dict().keys())
hit = [k for k in sd if k in own]
if len(hit) != len(sd):
    raise SystemExit(f'키 불일치: {len(hit)}/{len(sd)} 만 맞음. 예: {[k for k in sd if k not in own][:2]}')
missing, unexpected = dit.load_state_dict(sd, strict=False)
assert not unexpected, unexpected
# 실제로 모델에 실렸는지 모델 쪽에서 다시 확인
wm = torch.cat([p.detach().float().flatten() for n, p in dit.named_parameters() if n.endswith('.B')])
log(f'LoRA 텐서 {len(hit)}/{len(sd)}개 로드 | 모델의 B 가중치 absmax {wm.abs().max():.4f}, std {wm.std():.5f}')
if wm.abs().max() == 0: raise SystemExit('가중치가 0이다. 로드 실패.')

# 입력: 배경 제거 필수
IN = Path(a.input)
imgs = {}
for k in ['front','left','back','right']:
    f = IN/f'{k}.png'
    if f.exists(): imgs[k] = remove_background(Image.open(f), 'auto')
log(f'입력 뷰 {list(imgs)}')
for k, v in imgs.items():
    al = np.asarray(v)[...,3]
    log(f'  {k}: 투명 비율 {np.mean(al==0):.2f}')

def set_scale(s):
    for m in dit.modules():
        if isinstance(m, LoRALinear): m.scale = s * (m.alpha / m.r)

meshes, titles = [], []
for sc in [float(x) for x in a.scales.split(',')]:
    set_scale(sc)
    log(f'생성 중: LoRA scale {sc}')
    t0 = time.time()
    with torch.no_grad():
        m = pipe(image=imgs, num_inference_steps=a.steps, octree_resolution=a.octree,
                 num_chunks=20000, generator=torch.manual_seed(a.seed), output_type='trimesh')[0]
    tag = 'base' if sc == 0 else f'lora{sc:g}'
    m.export(OUT/f'{tag}.glb')
    log(f'  {tag}: faces {len(m.faces)}, {time.time()-t0:.0f}s -> {OUT}/{tag}.glb')
    meshes.append(m); titles.append(f'{tag} (scale {sc:g})')

def npts(m, n=40000, seed=0):
    # 시드를 고정하지 않으면 같은 메시끼리도 chamfer 가 0.01 쯤 나온다 (샘플링 잡음)
    v,_ = trimesh.sample.sample_surface(m, n, seed=seed)
    v = np.asarray(v)-m.bounds.mean(0); return v/np.abs(v).max()
if len(meshes) >= 2:
    A, B = npts(meshes[0]), npts(meshes[-1])
    ch = (cKDTree(B).query(A)[0].mean()+cKDTree(A).query(B)[0].mean())/2
    same = np.allclose(meshes[0].vertices, meshes[-1].vertices) if meshes[0].vertices.shape == meshes[-1].vertices.shape else False
    log(f'base vs LoRA chamfer: {ch:.5f} | 정점 완전 동일: {same}')

# 렌더
def rotY(P, deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    return P @ np.array([[c,0,s],[0,1,0],[-s,0,c]]).T
light = np.array([0.3,0.5,0.8]); light /= np.linalg.norm(light)
views = [('front',(0,1,2),1), ('back',(0,1,2),-1), ('left side',(2,1,0),-1), ('3/4',None,None)]
fig, axs = plt.subplots(len(meshes), 4, figsize=(15,4*len(meshes)), facecolor='#333')
axs = np.atleast_2d(axs)
for r, (m, t) in enumerate(zip(meshes, titles)):
    mm = m.simplify_quadric_decimation(face_count=200000) if len(m.faces) > 200000 else m
    V = np.asarray(mm.vertices); V = V - mm.bounds.mean(0); V = V/np.abs(V).max()
    N = mm.face_normals
    for c, (vt, ax_, flip) in enumerate(views):
        ax = axs[r,c]; P, NN = V, N
        if ax_ is None: P, NN = rotY(V,-35), rotY(N,-35); hx,vy,dz,flip = 0,1,2,1
        else: hx, vy, dz = ax_
        tri = P[mm.faces]; order = np.argsort(tri[:,:,dz].mean(1)*flip)
        sh = np.clip(np.abs(NN@light), 0.15, 1)
        ax.add_collection(PolyCollection(tri[order][:,:,[hx,vy]],
                          facecolors=plt.cm.gray(sh[order]), edgecolors='none', antialiased=False))
        ax.set_xlim(-1.05,1.05); ax.set_ylim(-1.05,1.05); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title((t+' | ' if c==0 else '')+vt, color='w', fontsize=10, loc='left')
plt.tight_layout(); plt.savefig(OUT/'compare.png', dpi=72, facecolor='#333')
log(f'비교 이미지 저장: {OUT}/compare.png')
