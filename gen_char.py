#!/usr/bin/env python
"""
SD(치비) 애니 캐릭터 정면 이미지 생성기. 3D 파이프라인의 첫 단계.

  gen_char.py  →  front.png  →  turnaround.py (6뷰)  →  generate.py (GLB)

베이스: cagliostrolab/animagine-xl-4.0 (로컬 캐시, 애니 SDXL)
LoRA : /data/sungju030307/loras/<이름>/*.safetensors  (--lora 로 지정, 여러 개 가능)

사용 예:
  python gen_char.py --prompt "1girl, blue hair, cat ear hood, pajamas" --out outputs/char1
  python gen_char.py --prompt "..." --lora sjchibi:0.6 --n 4 --seed 7      # 직접 학습한 화풍 (권장 0.6)
  python gen_char.py --prompt "..." --lora sjchibi/checkpoint-900:0.6     # 중간 체크포인트

실측 메모 (2026-09-28, sjchibi 900step):
  - 세기 0.8 은 배경에 발자국/반짝이가 떠서 3D 입력에 부적합. 0.6 + NEG 보강이면 깨끗.
  - 프롬프트 소품 단어(paw print ...)가 배경으로 샌다. 'cat slippers' 처럼 물건 이름으로 쓸 것.
  - CLIP 77 토큰 한도: 캐릭터 묘사는 30 토큰 안쪽으로. 넘치면 뒤의 품질 태그가 잘린다(무해).
  python gen_char.py --prompt "..." --no_chibi          # 치비 태그 없이
  python gen_char.py --prompt "..." --stand             # 앉은 자세 대신 서 있음

출력: <out>/front_<seed>.png (+ grid.png). 3D 로 넘길 땐 하나 골라 front.png 로 복사.
"""
import argparse, os, sys, time, glob
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--prompt', required=True, help='캐릭터 묘사 (danbooru 태그 스타일이 잘 먹음)')
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=4, help='생성 장수 (시드 순차 증가)')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=28)
ap.add_argument('--cfg', type=float, default=6.0)
ap.add_argument('--size', type=int, default=1024)
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--base', default='cagliostrolab/animagine-xl-4.0')
ap.add_argument('--lora', action='append', default=[],
                help='이름:세기  예) kawaii_chibi:0.8  (/data/sungju030307/loras/<이름>/ 의 safetensors). 반복 가능')
ap.add_argument('--no_chibi', action='store_true')
ap.add_argument('--stand', action='store_true')
ap.add_argument('--neg', default=None, help='네거티브 덮어쓰기')
ap.add_argument('--palette', type=int, default=0,
                help='N>0 이면 생성 후 전경을 색조 계열별 N색 무광으로 양자화 (외곽선 검정은 별도 유지). '
                     '디퓨전은 프롬프트로 색 수를 못 지키므로 후처리로 확정한다. 예: --palette 5')
ap.add_argument('--colors', default=None,
                help='팔레트를 직접 지정 "#피부,#흰,#하늘,#분홍,#검정" — 있으면 --palette 대신 이 색으로 양자화')
a = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
LORA_ROOT = Path('/data/sungju030307/loras')
def log(*m): print(time.strftime('[%H:%M:%S]'), *m, flush=True)

# ── 프롬프트: 3D 재구성에 유리한 조건을 고정으로 붙인다 ─────────────────────
# (정면, 전신, 흰 배경, 그림자 없음, 평평한 조명, 소품 없음)
# CLIP 은 77 토큰까지만 읽고 뒤는 버린다. 그래서 고정 태그는 짧게, 중요한 것부터.
CHIBI = 'chibi, super deformed, big head, '
POSE_SIT = 'sitting, legs folded, '
POSE_STAND = 'standing, a-pose, '
FIXED = ('solo, full body, front view, facing viewer, centered, white background, '
         'flat color, cel shading, no shadow, masterpiece, best quality')
# 'paw print, pattern, sparkles, floating objects' 는 sjchibi LoRA 를 켰을 때 프롬프트의 소품이
# 배경 장식으로 새는 걸 막는다 (실측: 이거 없으면 발자국이 배경에 뜸). 3D 입력엔 항상 필요.
NEG = ('paw print, pattern, sparkles, floating objects, floor, ground, chair, furniture, '
       'lowres, bad anatomy, bad hands, multiple views, 2girls, text, watermark, '
       'cropped, from side, from behind, shadow, gradient background, scenery, '
       'worst quality, low quality')

# LoRA 별 트리거 단어. 프롬프트에 자동으로 앞에 붙인다.
TRIGGER = {'kawaii_chibi': 'kawaii chibi avatar', 'sjchibi': 'sjchibi'}

def build():
    p = ''
    for n in names:
        key = n.split('/')[0]          # 'sjchibi/checkpoint-300' 같은 하위 경로도 트리거는 같음
        if key in TRIGGER: p += TRIGGER[key] + ', '
    # 배경 조건을 맨 앞에. 학습 캐릭터 밖의 새 캐릭터에서 LoRA 가 흰 배경을 놓치는 걸 실측했다
    # (여우 소년: 회색 배경/의자 등장). 앞쪽 토큰이 더 세게 먹으므로 트리거 바로 뒤에 둔다.
    p += 'white background, simple background, '
    if not a.no_chibi: p += CHIBI
    p += POSE_STAND if a.stand else POSE_SIT
    return p + a.prompt.strip().rstrip(',') + ', ' + FIXED

# ── 파이프라인 ──────────────────────────────────────────────────────────────
log(f'베이스 로드: {a.base}')
pipe = StableDiffusionXLPipeline.from_pretrained(a.base, torch_dtype=torch.float16, use_safetensors=True)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

names, scales = [], []
for spec in a.lora:
    name, _, sc = spec.partition(':')
    files = sorted(glob.glob(str(LORA_ROOT / name / '*.safetensors')))
    if not files:
        log(f'경고: LoRA 없음 -> {LORA_ROOT/name}'); continue
    # kohya 포맷 LoRA 의 text-encoder 키(lora_te1_/lora_te2_)를 diffusers 0.40 이 못 넘겨
    # IndexError 를 낸다. UNet 키만 골라 넣는다 — 스타일은 거의 UNet 에 있어 실용상 차이 없음.
    from safetensors.torch import load_file
    sd = load_file(files[0]); unet_sd = {k: v for k, v in sd.items() if k.startswith('lora_unet_')}
    pipe.load_lora_weights(unet_sd if unet_sd else sd, adapter_name=name)
    names.append(name); scales.append(float(sc) if sc else 1.0)
    log(f'LoRA {name} ({Path(files[0]).name}) 세기 {scales[-1]}')
if names:
    pipe.set_adapters(names, scales)

prompt, neg = build(), (a.neg if a.neg is not None else NEG)
log(f'prompt: {prompt[:140]}...')
imgs = []
for i in range(a.n):
    s = a.seed + i; t0 = time.time()
    im = pipe(prompt=prompt, negative_prompt=neg, num_inference_steps=a.steps,
              guidance_scale=a.cfg, width=a.size, height=a.size,
              generator=torch.Generator('cuda').manual_seed(s)).images[0]
    if a.colors or a.palette > 0:
        from palette import fit_palette, apply_palette, swatch, parse_colors, THRESH
        THRESH.update(cool=11.0, warm=5.0)   # 실측 확정값 (채도 높인 프롬프트 + 고정 문턱이 가장 정확)
        pal = parse_colors(a.colors) if a.colors else fit_palette(im, a.palette); im.save(OUT / f'front_{s}_raw.png')
        im = apply_palette(im, pal); swatch(pal).save(OUT / f'front_{s}_palette.png')
        log(f'seed {s}: 팔레트 {["#%02x%02x%02x" % tuple(int(x) for x in c) for c in pal]}')
    im.save(OUT / f'front_{s}.png'); imgs.append(im)
    log(f'seed {s}: {time.time()-t0:.0f}s -> {OUT}/front_{s}.png')

# 고르기 쉽게 그리드 한 장
if len(imgs) > 1:
    c = min(4, len(imgs)); r = (len(imgs) + c - 1) // c; T = 384
    g = Image.new('RGB', (T*c, T*r), 'white')
    for k, im in enumerate(imgs): g.paste(im.resize((T, T)), ((k % c)*T, (k // c)*T))
    g.save(OUT / 'grid.png'); log(f'grid -> {OUT}/grid.png')
log('완료. 마음에 드는 걸 front.png 로 복사한 뒤 turnaround.py 로 넘기면 됨')
