#!/usr/bin/env python
"""
정면 이미지 1장 -> 앉은 SD(치비) 4뷰 턴어라운드 (front/back/left/right)

Qwen-Image-Edit-2509 (지시 기반 편집) 로 뷰를 순차 생성한다.

핵심 두 가지:
 1) few-shot 조건: 뷰를 하나 만들 때마다 그것을 다음 뷰의 참조 이미지로 같이 넣는다.
    파이프라인이 이미지 리스트를 받아 프롬프트에서 "Picture 1/2/3" 로 지칭하므로,
    프롬프트에서 각 장이 무엇인지 명시해 준다. 원본 한 장만 넣으면 뷰마다
    머리 모양/의상이 달라져서 4뷰 3D 재구성이 실패한다.
 2) 4 GPU 분산: 20B 모델이라 3090 한 장(24GB)에 안 올라간다. cpu offload 는
    너무 느려서(5분에 1장도 못 뽑음) device_map='balanced' 로 4장에 나눠 올린다.

사용 예:
  python turnaround.py --front inputs/sample1_character/front.png --out outputs/turn1
  python turnaround.py --front a.png --out outputs/turn1 --no_chibi   # 뷰 회전만
  python turnaround.py --front a.png --out outputs/turn1 --anime_scale 0

출력: <out>/{front,back,left,right}.png  (generate.py --input_dir 에 바로 사용 가능)
"""
import argparse, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--front', required=True)
ap.add_argument('--out',   required=True)
ap.add_argument('--gpus',  default='2,3', help="분산에 쓸 GPU (2장이면 충분, 3장은 2배 빠름). 한 장만 쓰려면 '2'")
ap.add_argument('--steps', type=int, default=8, help='Lightning LoRA 기준 8')
ap.add_argument('--cfg',   type=float, default=1.0, help='Lightning 은 1.0 고정')
ap.add_argument('--seed',  type=int, default=1234)
ap.add_argument('--size',  type=int, default=1024)
ap.add_argument('--anime_scale', type=float, default=0.0,
                help='애니 LoRA 세기. 기본 0(끔): 이 LoRA 는 2511 용이라 2509 에서 검증 안 됨')
ap.add_argument('--lightning', action='store_true',
                help='Lightning 8step LoRA 사용. 기본 끔: 실측(2026-09-29) 8step=검은화면, 30step=노이즈. '
                     '베이스 모델만으로 8step/cfg1 이 정상 동작하므로 굳이 필요 없음')
ap.add_argument('--views', default='front,back',
                help='생성할 뷰 (쉼표, 생성 순서). 기본 앞/뒤 2장. 실측(2026-09-29): 옆모습(2단계 생성)을 '
                     '넣은 3뷰는 면 수는 늘지만 옆모습의 포즈/크기가 정면과 미세하게 어긋나 형상이 더 뭉개짐. '
                     '옆모습이 정말 필요하면 front,back,left (4장: front,back,left,right)')
ap.add_argument('--side_tries', type=int, default=3, help='옆모습 2단계(앉히기) 최대 시도 횟수 (side_mode=twopass 일 때)')
ap.add_argument('--side_mode', choices=['fewshot', 'twopass'], default='fewshot',
                help="옆모습 방법. fewshot: 앉은 정면+뒷모습 2장을 참조로 '3D 프린팅용 정사영 턴어라운드의 빠진 옆모습' 한 번에 생성(2026-09-30 검증, 기본). twopass: 서 있는 옆모습→앉히기")
ap.add_argument('--max_mem', type=float, default=0,
                help='GPU 당 모델 적재 상한(GiB). balanced 가 한 장에 몰아 넣어 1024 에서 멈추는 걸 막는다. 예: 18')
ap.add_argument('--max_refs', type=int, default=3,
                help='한 번에 넣을 참조 이미지 수. Qwen 권장 상한이 3이다')
ap.add_argument('--palette', type=int, default=0,
                help='N>0 이면 완성된 모든 뷰를 front 에서 뽑은 **같은** N색 팔레트로 양자화 (뷰 간 색 일치)')
ap.add_argument('--colors', default=None, help='팔레트 직접 지정 "#..,#..,.." (있으면 --palette 대신)')
ap.add_argument('--flat_desc', default=None,
                help='모든 뷰 프롬프트에 붙일 색/스타일 서술. 5색 양자화를 쓸 땐 필수에 가까움 — 옅게 그려지면 '
                     '후드/파자마가 못 갈림(실측). --colors 만 주면 기본값: flat solid colors, saturated sky blue '
                     'hair and pajamas, pure white hood, pink accents')
ap.add_argument('--no_chibi', action='store_true')
ap.add_argument('--no_sit',   action='store_true')
a = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus
os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')

import torch
import numpy as np
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
def log(*m): print(time.strftime('[%H:%M:%S]'), *m, flush=True)

HC = '/data/sungju030307/hf-cache'
LIGHTNING = (f'{HC}/models--lightx2v--Qwen-Image-Lightning/snapshots/'
             'e74da8d4e71a54b341de86aa9f8d2509165aa513/Qwen-Image-Edit-2509/'
             'Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors')
ANIME = (f'{HC}/models--prithivMLmods--Qwen-Image-Edit-2511-Anime/snapshots/'
         '977f8c2ed22c8581c8750370bfce34058b253598/'
         'Qwen-Image-Edit-2511-Anime-2000.safetensors')

# ── 공통 조건 ───────────────────────────────────────────────────────────────
# 3D 재구성용이라 "무엇을 유지할지" 가 "무엇을 바꿀지" 보다 중요하다.
STYLE = ('Anime cel shading with clean line art. '
         'Pure solid white background, no floor, no props, no cast shadow. '
         'Flat even lighting from the front. '
         'Orthographic camera at the character eye level, no perspective distortion. '
         'The whole body fits in frame with a small even margin, centered, not cropped.')
IDENTITY = ('Keep the character design exactly identical: same hairstyle and hair length, '
            'same hair color, same eye color, same face, same outfit and its colors, '
            'same accessories, same body proportions and same overall size in frame.')
CHIBI = ('super-deformed chibi proportions, about two heads tall, '
         'large round head, small simple body, ')
SIT = 'sitting on the ground with both legs folded in front, arms relaxed at the sides, '
STAND = 'standing upright, arms slightly away from the body, '

NEG = ('multiple characters, two figures, side-by-side panels, grid, collage, '
       'text, caption, watermark, signature, cropped body, cut off limbs, '
       'different character, different outfit, different hairstyle, '
       'cast shadow, ground shadow, floor, furniture, background scenery, '
       'perspective distortion, tilted camera, close-up')

VIEW_DESC = {
    'front': 'seen from the FRONT, facing the camera directly',
    # 옆모습은 앉은 포즈를 강조하면 정면으로 되돌아가는 경향(실측). 옆모습 단서를 여러 개 겹쳐 준다.
    'left':  ('seen from the LEFT SIDE in strict 90 degree profile: the face is in profile, only one eye '
              'is visible, the nose points to the left edge of the image, the back of the head is on the right'),
    'back':  'seen from BEHIND, a 180 degree back view showing the back of the head and the back of the outfit, the face is not visible',
    'right': ('seen from the RIGHT SIDE in strict 90 degree profile: the face is in profile, only one eye '
              'is visible, the nose points to the right edge of the image, the back of the head is on the left'),
}

def pose():
    s = CHIBI if not a.no_chibi else ''
    return s + (STAND if a.no_sit else SIT)

FLAT = a.flat_desc
if FLAT is None and a.colors:
    FLAT = ('flat solid colors with no gradients and no shading, saturated sky blue hair and pajamas, '
            'pure white cat ear hood, pink accents')
if FLAT:
    STYLE = FLAT + '. ' + STYLE

def build_prompt(view, refs):
    """refs = [(라벨, 이미지), ...] 순서가 Picture 1,2,3 에 대응한다.
    실측(2026-09-29): 긴 프롬프트(정체성/스타일 조건을 뒤에 길게)로는 cfg 1 에서 회전 지시가
    묻혀 세 장 다 정면이 나왔다. 진단에서 통했던 짧고 직접적인 문장(회전 지시를 맨 앞에)으로 간다."""
    n = len(refs)
    if view == 'front':
        return (f'Redraw the character in Picture 1 as a {pose()}{VIEW_DESC["front"]}. '
                f'Keep the same hairstyle, hair color, eyes, outfit and colors. '
                f'Plain white background, anime cel shading, no shadow, full body centered.')
    # 포즈를 빼면 옆/뒤 뷰가 서 있는 자세로 바뀌고 크기도 달라진다(실측). 뷰마다 포즈를 반복 명시.
    return (f'Redraw the same character from Picture {n}, {pose()}now {VIEW_DESC[view]}. '
            f'Keep the exact same pose and the same size in frame as Picture {n}. '
            f'Same character, same hairstyle, same outfit and colors. '
            f'Plain white background, anime cel shading, no shadow, full body centered.')

# ── 파이프라인 ──────────────────────────────────────────────────────────────
ng = len(a.gpus.split(','))
log(f'파이프라인 로드 (54GB, GPU {ng}장 분산)')
try:
    kw = {}
    if a.max_mem > 0:
        kw['max_memory'] = {i: f'{a.max_mem}GiB' for i in range(ng)}
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        'Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.bfloat16, device_map='balanced', **kw)
    mode = f'{ng}장 분산' + (f' (장당 {a.max_mem}GiB 상한)' if a.max_mem > 0 else '')
except Exception as e:
    log(f'분산 로드 실패({type(e).__name__}) -> cpu offload 로 대체')
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        'Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload(); mode = 'cpu offload'

ads = []
if a.lightning and Path(LIGHTNING).exists():
    pipe.load_lora_weights(LIGHTNING, adapter_name='lightning'); ads.append(('lightning', 1.0))
if a.anime_scale > 0 and Path(ANIME).exists():
    pipe.load_lora_weights(ANIME, adapter_name='anime'); ads.append(('anime', a.anime_scale))
if ads:
    pipe.set_adapters([n for n, _ in ads], [s for _, s in ads])

def align_lora_devices(root):
    """device_map 으로 모듈이 여러 GPU 에 흩어진 상태에서 load_lora_weights 를 하면
    LoRA 텐서만 CPU 에 남아 'two devices cuda:0 and cpu' 로 죽는다.
    각 LoRA 레이어를 자기 base_layer 와 같은 장치/dtype 으로 맞춰 준다."""
    n = 0
    for m in root.modules():
        base = getattr(m, 'base_layer', None)
        if base is None: continue
        w = getattr(base, 'weight', None)
        if w is None: continue
        for attr in ('lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B'):
            d = getattr(m, attr, None)
            if d is None: continue
            for k in list(d.keys()):
                if hasattr(d[k], 'to'):
                    d[k] = d[k].to(device=w.device, dtype=w.dtype); n += 1
    return n

if ads:
    for comp in (pipe.transformer, getattr(pipe, 'text_encoder', None)):
        if comp is not None:
            moved = align_lora_devices(comp)
            if moved: log(f'LoRA 텐서 {moved}개 장치 정렬')
log(f'{mode} | LoRA: {", ".join(f"{n}({s})" for n, s in ads) or "없음"}')

src = Image.open(a.front).convert('RGB')
log(f'입력 {a.front} {src.size}')

# ── 순차 생성 (앞서 만든 뷰를 다음 뷰의 참조로 넘긴다) ──────────────────────
made = {}
LABEL = {'front': 'the same character in chibi form, front view',
         'left':  'the same character in chibi form, left side view',
         'back':  'the same character in chibi form, back view'}

VIEWS_ORDER = [v.strip() for v in a.views.split(',') if v.strip()]
if 'front' not in VIEWS_ORDER: VIEWS_ORDER.insert(0, 'front')   # front 는 다른 뷰의 기준이라 필수
VIEWS_ORDER = ['front'] + [v for v in VIEWS_ORDER if v != 'front']
if a.side_mode == 'fewshot' and any(v in VIEWS_ORDER for v in ('left', 'right')):
    if 'back' not in VIEWS_ORDER: VIEWS_ORDER.insert(1, 'back')   # 옆모습 참조로 뒷모습 필요
    VIEWS_ORDER = [v for v in VIEWS_ORDER if v not in ('left', 'right')] + [v for v in VIEWS_ORDER if v in ('left', 'right')]
for view in VIEWS_ORDER:
    if view == 'front':
        refs = [('the original character reference', src)]
    else:
        # 원본 + 지금까지 만든 뷰 중 최신 것들. 마지막이 치비 정면(정체성 기준)이 되도록 배치.
        pool = [('the original character reference', src)]
        for v in ['left', 'back']:
            if v in made and v != view: pool.append((LABEL[v], made[v]))
        pool.append((LABEL['front'], made['front']))
        refs = pool[-a.max_refs:]
    prompt = build_prompt(view, refs)
    t0 = time.time()
    if view in ('left', 'right') and a.side_mode == 'fewshot':
        # 사용자 제안(2026-09-30): 앉은 정면+뒷모습을 few-shot 으로 주고 '3D 프린팅 모델링용 정사영 턴어라운드 시트의
        # 빠진 옆모습'을 요청 → 한 번에 앉은 옆모습(비례·디자인 유지). 2단계 방식보다 안정적(3/3 성공).
        sd = 'LEFT' if view == 'left' else 'RIGHT'; near, far = ('left', 'right') if view == 'left' else ('right', 'left')
        p = (f'Picture 1 is the front view and Picture 2 is the back view of the same chibi character, drawn as an orthographic '
             f'character turnaround sheet for 3D printing modeling: same scale, same pose, same proportions in every view. '
             f'Draw the missing {sd} SIDE view of this turnaround: a strict 90 degree profile, the face in profile with only one eye '
             f'visible, the nose pointing to the {near} edge, the back of the head on the {far}. '
             + ('The character is seated on the ground with legs folded, exactly as in Picture 1 and Picture 2, so the body is a low compact shape. ' if not a.no_sit else 'Keep exactly the same pose as in Picture 1 and Picture 2. ')
             + f'Same character, same hairstyle, same hood, same outfit and colors, same head size and same height in frame as Picture 1. '
             f'Plain white background, anime cel shading, no shadow, full body centered.')
        neg = NEG + ', front view, face turned to camera, both eyes visible'
        img = pipe(image=[made['front'], made['back']], prompt=p, negative_prompt=neg,
                   num_inference_steps=a.steps, true_cfg_scale=a.cfg, height=a.size, width=a.size,
                   generator=torch.Generator(device='cuda').manual_seed(a.seed)).images[0]
        log(f'{view}: few-shot(앞+뒤 참조) {time.time()-t0:.0f}s')
    elif view in ('left', 'right') and not a.no_sit:
        # 옆모습 + 앉은 자세를 한 번에 요구하면 정면으로 되돌아간다(4회 실측). 두 단계로:
        #  1) 포즈 언급 없이 옆모습만 -> 서 있는 옆모습
        #  2) 그 한 장만 참조로(앉은 정면을 넣으면 복제해 버림), "sitting" 단어 없이 자세를 기하학적으로 서술
        side = 'left' if view == 'left' else 'right'
        # 비례: 옆모습이 정면보다 머리가 작게(덜 치비로) 나오면 3D 형상이 어긋나므로 치비 비례를 명시(2026-09-29)
        p1 = (f'Redraw the same character from Picture {len(refs)}, now {VIEW_DESC[view]}. '
              f'{CHIBI}the head is as large relative to the body as in Picture {len(refs)}. '
              f'Same character, same hairstyle, same outfit and colors. '
              f'Plain white background, anime cel shading, no shadow, full body centered.')
        stand = pipe(image=[im for _, im in refs], prompt=p1, negative_prompt=NEG,
                     num_inference_steps=a.steps, true_cfg_scale=a.cfg, height=a.size, width=a.size,
                     generator=torch.Generator(device='cuda').manual_seed(a.seed)).images[0]
        stand.save(OUT/f'{view}_standing.png')
        p2 = (f'Keep this exact {side} side profile view of the character and the same design. Change only the '
              f'legs: bend the knees and fold the legs so the character rests on the ground, the body becomes a '
              f'low compact shape, the head stays at the top. Plain white background, anime cel shading.')
        # 768 에서는 2단계가 서 있는 채로 나오기도 함(2026-09-29 실측) → 바운딩박스 비율로 '앉음' 판정, 실패 시 시드 바꿔 재시도
        def hw(im):
            g = np.asarray(im.convert('RGB')).min(-1) < 235; ys, xs = np.where(g)
            return (ys.max() - ys.min() + 1) / max(1, xs.max() - xs.min() + 1)
        r0 = hw(stand); img = None
        for k in range(a.side_tries):
            cand = pipe(image=[stand], prompt=p2, negative_prompt=NEG,
                        num_inference_steps=a.steps, true_cfg_scale=max(a.cfg, 3.5), height=a.size, width=a.size,
                        generator=torch.Generator(device='cuda').manual_seed(a.seed + k)).images[0]
            r = hw(cand); cand.save(OUT/f'{view}_try{k}.png')
            log(f'{view}: 2단계 시도 {k} 세로/가로 {r:.2f} (서 있음 {r0:.2f})')
            if r < 0.85 * r0: img = cand; break
        if img is None: img = cand; log(f'{view}: 앉은 자세 판정 실패 — 마지막 시도 사용')
        log(f'{view}: 2단계(옆모습→앉은 자세) {time.time()-t0:.0f}s')
    else:
        img = pipe(image=[im for _, im in refs], prompt=prompt, negative_prompt=NEG,
                   num_inference_steps=a.steps, true_cfg_scale=a.cfg,
                   height=a.size, width=a.size,
                   generator=torch.Generator(device='cuda').manual_seed(a.seed)).images[0]
    made[view] = img; img.save(OUT/f'{view}.png')
    log(f'{view}: 참조 {len(refs)}장, {time.time()-t0:.0f}s -> {OUT}/{view}.png')

if a.colors or a.palette > 0:
    # 뷰마다 따로 양자화하면 색이 미세하게 어긋난다. front 에서 팔레트를 한 번 뽑아(또는 지정 색) 전 뷰에 공유.
    from palette import fit_palette, apply_palette, swatch, parse_colors, THRESH
    THRESH.update(cool=11.0, warm=5.0)   # 실측 확정값, 모든 뷰에 공유
    pal = parse_colors(a.colors) if a.colors else fit_palette(made['front'], a.palette)
    for v in VIEWS_ORDER:
        made[v].save(OUT / f'{v}_raw.png'); made[v] = apply_palette(made[v], pal); made[v].save(OUT / f'{v}.png')
    swatch(pal).save(OUT / 'palette.png')
    log(f'{len(pal)}색 공유 팔레트 적용: {["#%02x%02x%02x" % tuple(int(x) for x in c) for c in pal]}')

# 확인용 시트
n = len(VIEWS_ORDER)
sheet = Image.new('RGB', (a.size*n, a.size), 'white')
for i, v in enumerate(VIEWS_ORDER):
    sheet.paste(made[v], (i*a.size, 0))
sheet.resize((a.size//2*n, a.size//2)).save(OUT/'sheet.png')
log(f'완료. generate.py --input_dir {OUT} 로 3D 생성 가능')
