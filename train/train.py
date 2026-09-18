#!/usr/bin/env python
"""
Hunyuan3D-2mv Shape DiT LoRA 학습 (노트북과 동일한 로직, 터미널 분리 실행용).

  # 단일 GPU
  nohup python train/train.py --epochs 60 --batch 8 > .../train.log 2>&1 &

  # 4장 분산 (LoRA 그래디언트가 4MB 뿐이라 거의 선형으로 확장된다)
  nohup torchrun --nproc_per_node=4 train/train.py --epochs 60 --batch 8 > .../train.log 2>&1 &

ckpt/lora_last.pt 가 있으면 자동으로 이어서 학습한다.
분산 실행 시 --batch 는 GPU 한 장당 배치이고, 실제 배치는 batch x 4 가 된다.
"""
import argparse, os, sys, json, math, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--cache',  default='/data/sungju030307/img2glb/train/cache')
ap.add_argument('--ckpt',   default='/data/sungju030307/img2glb/train/ckpt')
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--batch',  type=int, default=8)
ap.add_argument('--lr',     type=float, default=1e-4)
ap.add_argument('--rank',   type=int, default=8)
ap.add_argument('--alpha',  type=int, default=16)
ap.add_argument('--targets', default='attn', choices=['attn','attn+mlp'])
ap.add_argument('--drop_cond', type=float, default=0.1)
ap.add_argument('--ckpt_every', type=int, default=250)
ap.add_argument('--gpu',    type=int, default=0)
ap.add_argument('--seed',   type=int, default=0)
ap.add_argument('--no_resume', action='store_true')
a = ap.parse_args()

# torchrun 으로 띄우면 RANK/WORLD_SIZE 가 환경변수로 들어온다
RANK       = int(os.environ.get('RANK', -1))
WORLD      = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
DDP_ON     = RANK >= 0
if not DDP_ON:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
REPO = Path('/data/sungju030307/img2glb/Hunyuan3D-2')
sys.path.insert(0, str(REPO)); sys.path.insert(0, '/home/sungju030307/img2glb')

import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

if DDP_ON:
    import torch.distributed as dist
    dist.init_process_group('nccl')
    torch.cuda.set_device(LOCAL_RANK)
    DEV = f'cuda:{LOCAL_RANK}'
else:
    DEV = 'cuda:0'
IS_MAIN = (not DDP_ON) or RANK == 0
CKPT = Path(a.ckpt); CKPT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed + max(RANK,0)); np.random.seed(a.seed + max(RANK,0))
def log(*m):
    if IS_MAIN: print(time.strftime('[%H:%M:%S]'), *m, flush=True)
log(f'분산 학습 {WORLD}장' if DDP_ON else '단일 GPU')

log('모델 로드')
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv', subfolder='hunyuan3d-dit-v2-mv', variant='fp16', device=DEV)
dit, conditioner, sched = pipe.model, pipe.conditioner, pipe.scheduler
DT = next(dit.parameters()).dtype          # dit.dtype 은 없다
for p in dit.parameters(): p.requires_grad_(False)
for p in conditioner.parameters(): p.requires_grad_(False)

# ★ LoRA A/B 는 fp32. fp16 이면 AdamW eps 언더플로로 1스텝 만에 NaN.
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, attn_out=None):
        super().__init__(); self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        dv = base.weight.device; self.out_slice = attn_out
        of = attn_out or base.out_features
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dv, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(of, r, device=dv, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.r, self.alpha, self.scale = r, alpha, alpha/r
    def forward(self, x):
        # 가중치는 fp32 로 보관(AdamW 안정성)하되 연산은 base dtype 으로.
        # x.float() 로 올리면 활성값 전체가 fp32 가 되어 4배 느려진다. A/B 는 작으므로
        # 그쪽을 내리는 편이 훨씬 싸다. autograd 가 fp32 파라미터로 그래디언트를 돌려준다.
        y = self.base(x)
        A = self.A.to(x.dtype); B = self.B.to(x.dtype)
        d = (x @ A.T) @ B.T * self.scale
        if self.out_slice:
            y = y.clone(); y[..., :self.out_slice] = y[..., :self.out_slice] + d; return y
        return y + d

hid, n = dit.hidden_size, 0
for blk in dit.double_blocks:
    for at in (blk.img_attn, blk.txt_attn):
        for nm in ('qkv','proj'): setattr(at, nm, LoRALinear(getattr(at,nm), a.rank, a.alpha)); n += 1
    if a.targets == 'attn+mlp':
        for mlp in (blk.img_mlp, blk.txt_mlp):
            for i in (0,2): mlp[i] = LoRALinear(mlp[i], a.rank, a.alpha); n += 1
for blk in dit.single_blocks:
    blk.linear1 = LoRALinear(blk.linear1, a.rank, a.alpha,
                             attn_out=(3*hid if a.targets == 'attn' else None)); n += 1
    blk.linear2 = LoRALinear(blk.linear2, a.rank, a.alpha); n += 1

class Ck(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, *ar, **kw):
        ks = list(kw); vs = list(ar)+[kw[k] for k in ks]; na = len(ar)
        return checkpoint(lambda *v: s.m(*v[:na], **dict(zip(ks, v[na:]))), *vs, use_reentrant=False)
dit.double_blocks = nn.ModuleList([Ck(b) for b in dit.double_blocks])
dit.single_blocks = nn.ModuleList([Ck(b) for b in dit.single_blocks])
lora = [p for p in dit.parameters() if p.requires_grad]
log(f'LoRA {n}개 Linear, {sum(p.numel() for p in lora)/1e6:.2f}M 파라미터')

net = dit
if DDP_ON:
    from torch.nn.parallel import DistributedDataParallel as DDP
    net = DDP(dit, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
              find_unused_parameters=False, gradient_as_bucket_view=True)
    net._set_static_graph()     # activation checkpointing 과 함께 쓸 때 필요
    log('DDP 래핑 완료')

class DS(Dataset):
    def __init__(s, c):
        s.items = [d for d in sorted(Path(c).iterdir())
                   if (d/'latent.npy').exists() and (d/'cond.npy').exists()]
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        d = s.items[i]
        return (torch.from_numpy(np.load(d/'latent.npy')).float(),
                torch.from_numpy(np.load(d/'cond.npy')).float())
ds = DS(a.cache)
if DDP_ON:
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True, drop_last=True)
    dl = DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=2,
                    drop_last=True, pin_memory=True, persistent_workers=True)
else:
    sampler = None
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True, persistent_workers=True)
steps_total = a.epochs * max(len(dl), 1)
log(f'샘플 {len(ds)}개 | GPU당 배치 {a.batch} (실효 {a.batch*WORLD}) | '
    f'에폭당 {len(dl)} 스텝 | 총 {steps_total} 스텝')

opt = torch.optim.AdamW(lora, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
schedlr = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps_total,1), pct_start=0.05)
start, hist = 0, []
if not a.no_resume and (CKPT/'lora_last.pt').exists():
    ck = torch.load(CKPT/'lora_last.pt', map_location='cpu', weights_only=False)
    dit.load_state_dict(ck['lora'], strict=False)
    start, hist = ck.get('step', 0), list(ck.get('hist', []))
    for _ in range(start): schedlr.step()
    log(f'이어서 학습: step {start} 부터')

UNCOND = None
def train_step(x0, cond):
    global UNCOND
    B = x0.shape[0]
    # ★ 이 레포는 시간 방향이 뒤집혀 있다. sigma 0 = 노이즈, sigma 1 = 깨끗.
    #   (scheduler 의 sigma_min=1.0, sigma_max=0.001 로 이름이 거꾸로 붙어 있고
    #    추론 중 sigma 가 0.001 -> 1.0 으로 올라간다)
    #   따라서  x_t = (1-s)*noise + s*x0,  target = x0 - noise  가 맞다.
    #   반대로 쓰면 sigma 0.5 에서 예측과 목표의 코사인유사도가 -0.30 이 된다 (실측).
    sig32 = torch.rand(B, device=DEV, dtype=torch.float32)     # 타임스텝은 fp32
    s = sig32.to(x0.dtype).view(B,1,1)
    noise = torch.randn_like(x0)
    x_t = (1-s)*noise + s*x0; target = x0 - noise
    if a.drop_cond > 0:
        m = torch.rand(B, device=DEV) < a.drop_cond
        if m.any():
            if UNCOND is None or UNCOND.shape[0] < B:
                with torch.no_grad():
                    UNCOND = conditioner.unconditional_embedding(
                        B, view_idxs=[(0,1,2,3)]*B)['main'].to(DEV, x0.dtype)
            cond = cond.clone(); cond[m] = UNCOND[:B][m]
    # 모델에는 0~1 정규화 시간을 넣는다. 파이프라인(Hunyuan3DDiTFlowMatchingPipeline)이
    # 스케줄러 t 를 num_train_timesteps 로 나눠서 넘기고(pipelines.py:749),
    # DiT 내부 timestep_embedding 이 다시 x1000 을 한다(hunyuan3ddit.py:48).
    # 여기서 *1000 을 하면 내부에서 백만 배가 되어 사전학습 시간 조건 범위를 완전히 벗어난다.
    pred = net(x_t, sig32, {'main': cond})
    return torch.nn.functional.mse_loss(pred.float(), target.float())

def save(step, loss):
    if not IS_MAIN: return
    sd = {k: v.detach().cpu() for k, v in dit.state_dict().items() if ('.A' in k or '.B' in k)}
    o = {'step': step, 'loss': loss, 'args': vars(a), 'lora': sd, 'hist': hist}
    torch.save(o, CKPT/f'lora_{step:06d}.pt'); torch.save(o, CKPT/'lora_last.pt')

step, t0, done = start, time.time(), False
dit.train()
for ep in range(a.epochs):
    if done: break
    if sampler is not None: sampler.set_epoch(ep)
    for x0, cond in dl:
        x0 = x0.to(DEV, DT, non_blocking=True); cond = cond.to(DEV, DT, non_blocking=True)
        loss = train_step(x0, cond); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(lora, 1.0)
        opt.step(); schedlr.step(); opt.zero_grad(set_to_none=True)
        hist.append(loss.item()); step += 1
        if step % 25 == 0:
            el = time.time()-t0; per = el/max(step-start, 1)
            log(f'step {step}/{steps_total} ({step/steps_total*100:.0f}%)  '
                f'loss {np.mean(hist[-25:]):.4f}  lr {schedlr.get_last_lr()[0]:.2e}  '
                f'gnorm {float(gn):.2f}  {per:.2f}s/step  '
                f'{a.batch*WORLD/per*3600:.0f} samples/h  남음 {per*(steps_total-step)/60:.0f}분  '
                f'peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB')
        if step % a.ckpt_every == 0: save(step, float(np.mean(hist[-25:])))
        if step >= steps_total: done = True; break
save(step, float(np.mean(hist[-25:])))
log(f'학습 완료. {step} 스텝, {(time.time()-t0)/60:.1f}분, 최종 loss {np.mean(hist[-25:]):.4f}')
if DDP_ON:
    dist.barrier(); dist.destroy_process_group()
