#!/usr/bin/env python
"""
turnaround.py 검은 화면 진단. Lightning LoRA / 스텝 수 / cfg 를 하나씩 분리한다.
(2026-09-28 1차 진단: LoRA 없음=정상, Lightning+어떤 스케줄러든=검정. 그러나 LoRA/steps/cfg 를 동시에 바꿔 원인 미분리)

  D: LoRA 없음 + 8step + cfg1        -> 8step/cfg1 자체가 문제인가
  E: Lightning + 30step + cfg4       -> LoRA 자체가 문제인가
  F: Lightning fuse_lora 후 8step    -> device_map 분산 + peft 런타임 문제인가
각 케이스에서 latent NaN 여부를 직접 본다 (완전 검정 = NaN 이 0 으로 클램프된 전형 증상).

  CUDA_VISIBLE_DEVICES=1,2,3 python bench/diag_qwen_lightning.py
"""
import os, time, numpy as np, traceback
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1,2,3')
os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

HC = '/data/sungju030307/hf-cache'
LIGHT = (f'{HC}/models--lightx2v--Qwen-Image-Lightning/snapshots/e74da8d4e71a54b341de86aa9f8d2509165aa513/'
         'Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors')
OUT = '/data/sungju030307/img2glb/outputs/_diag_qwen'; os.makedirs(OUT, exist_ok=True)
def log(*m): print(time.strftime('[%H:%M:%S]'), *m, flush=True)

log('로드')
pipe = QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.bfloat16, device_map='balanced')
src = Image.open('/home/sungju030307/img2glb/inputs/smaple4/front.png').convert('RGB')
P = ('Picture 1 is a character. Redraw the same character seen from the LEFT SIDE, 90 degree side profile. '
     'Keep the identical design. Plain white background, anime cel shading.')

def align(root):
    """device_map 분산 상태에서 load_lora_weights 하면 LoRA 텐서만 CPU 에 남는다. base 와 맞춘다."""
    n = 0
    for m in root.modules():
        b = getattr(m, 'base_layer', None)
        if b is None or not hasattr(b, 'weight'): continue
        for at in ('lora_A', 'lora_B'):
            d = getattr(m, at, None)
            if d:
                for k in list(d.keys()):
                    if hasattr(d[k], 'to'): d[k] = d[k].to(device=b.weight.device, dtype=b.weight.dtype); n += 1
    return n

def run(tag, steps, cfg):
    t0 = time.time()
    try:
        g = torch.Generator(device='cuda').manual_seed(0)
        lat = pipe(image=[src], prompt=P, num_inference_steps=steps, true_cfg_scale=cfg, height=512, width=512,
                   generator=g, output_type='latent').images
        nan = bool(torch.isnan(lat).any()); amax = float(lat.float().abs().max())
        g = torch.Generator(device='cuda').manual_seed(0)
        im = pipe(image=[src], prompt=P, num_inference_steps=steps, true_cfg_scale=cfg, height=512, width=512,
                  generator=g).images[0]
        a = np.asarray(im).astype(float); im.save(f'{OUT}/{tag}.png')
        log(f'{tag}: 밝기 {a.mean():6.1f} std {a.std():5.1f} | latent NaN={nan} absmax={amax:.3g} | '
            f'{time.time()-t0:.0f}s {"검은화면" if a.mean() < 5 else "정상"}')
    except Exception as e:
        log(f'{tag}: 예외 {type(e).__name__}: {str(e)[:120]}')

run('D_noLoRA_8step_cfg1', 8, 1.0)
pipe.load_lora_weights(LIGHT, adapter_name='lightning'); pipe.set_adapters(['lightning'], [1.0])
log('LoRA 텐서 정렬', align(pipe.transformer))
run('E_light_30step_cfg4', 30, 4.0)
try:
    pipe.fuse_lora(adapter_names=['lightning'], lora_scale=1.0); pipe.unload_lora_weights(); log('fuse_lora 완료')
    run('F_light_fused_8step_cfg1', 8, 1.0)
except Exception:
    log('fuse 실패:', traceback.format_exc().strip().splitlines()[-1][:120])
