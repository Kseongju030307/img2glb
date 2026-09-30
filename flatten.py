#!/usr/bin/env python
"""
원본 캐릭터 그림을 **디자인 그대로 유지한 채** 5색 무광 스타일로 바꾼다 (이미지 조건 편집).
텍스트로 새로 생성하면(gen_char) 후드 모양·안대·슬리퍼가 바뀌므로, 원본을 참조로 넣고
"색만 평탄하게" 편집한 뒤 팔레트를 씌운다.

  python flatten.py --img inputs/smaple4/front.png --out generated/... [--colors "#..."]
"""
import argparse, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--img', default=None, help='한 장')
ap.add_argument('--imgs', nargs='*', default=None,
                help='여러 장(예: 턴어라운드 front_raw back_raw). 같은 팔레트를 공유해서 뷰 간 색이 일치')
ap.add_argument('--out', required=True)
ap.add_argument('--gpus', default='2,3')
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--steps', type=int, default=12)
ap.add_argument('--cfg', type=float, default=3.0)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n', type=int, default=2, help='시드 몇 개')
ap.add_argument('--colors', default='#f2d3c2,#ffffff,#8ec6ff,#f3a6c4,#000000')
ap.add_argument('--desc', default='sky blue hair, sky blue pajamas, pure white cat ear hood, pink accents',
                help='색 배정 설명 — 캐릭터마다 바꿔 줄 것')
ap.add_argument('--descs', nargs='*', default=None, help='--imgs 와 같은 순서로 장별 설명(뷰/포즈 포함). 있으면 --desc 대신 사용')
a = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus
os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
sys.path.insert(0, '/home/sungju030307/img2glb')
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from palette import apply_palette, parse_colors, swatch, THRESH
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
def log(*m): print(time.strftime('[%H:%M:%S]'), *m, flush=True)
ngpu = len(a.gpus.split(','))
if ngpu >= 3:
    pipe = QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.bfloat16, device_map='balanced')
    log(f'bf16, GPU {ngpu}장 분산')
else:
    # 사용자 지시로 GPU 2,3 만 사용 가능. bf16 54GB 는 2장(48GB)에 안 들어감.
    # nf4 4-bit 는 출력이 노이즈로 깨짐(실측). int8 은 충실도가 높으므로: transformer int8(~20GB) -> GPU0,
    # text encoder bf16(~15GB) -> GPU1 (device_map 훅이 입력을 자동으로 옮겨 준다), VAE -> GPU0.
    assert ngpu == 2, '--gpus 에 2장(예: 2,3)을 주세요'
    from diffusers import BitsAndBytesConfig as DBnB, QwenImageTransformer2DModel
    from transformers import Qwen2_5_VLForConditionalGeneration
    tr = QwenImageTransformer2DModel.from_pretrained('Qwen/Qwen-Image-Edit-2509', subfolder='transformer',
                                                     quantization_config=DBnB(load_in_8bit=True), torch_dtype=torch.bfloat16, device_map={'': 0})
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen-Image-Edit-2509', subfolder='text_encoder',
                                                            torch_dtype=torch.bfloat16, device_map={'': 1})
    pipe = QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', transformer=tr, text_encoder=te, torch_dtype=torch.bfloat16)
    pipe.vae.to('cuda:0')
    log(f'int8 transformer GPU0 {torch.cuda.memory_allocated(0)/2**30:.1f}GiB | bf16 text encoder GPU1 {torch.cuda.memory_allocated(1)/2**30:.1f}GiB')
paths = a.imgs if a.imgs else [a.img]
assert paths and paths[0], '--img 또는 --imgs 필요'
# 실측(2026-09-29): 이 프롬프트로 디자인(후드·안대·땋은 머리·리본·장갑)은 잘 유지되나, 512 에서 상반신으로
# 잘려 나왔음 → 전신/동일 프레이밍을 명시하고 잘림을 네거티브에 추가.
def prompt(desc):
    return (f'Redraw Picture 1 as exactly the same character in exactly the same pose, viewpoint, framing and scale: full body from head to feet, '
            f'feet visible, same head shape, same hairstyle, same hood, same accessories, same outfit, {desc}. Change only the rendering style: '
            f'flat solid colors with no gradients and no shading, clean dark outlines. Plain white background.')
NEG = 'cropped, half body, close-up, different character, changed pose, gradient, soft shading, blur, extra objects, text'
THRESH.update(cool=11.0, warm=5.0); pal = parse_colors(a.colors); swatch(pal).save(OUT / 'palette.png')
for k, path in enumerate(paths):
    P = prompt(a.descs[k] if a.descs else a.desc)
    src = Image.open(path).convert('RGB'); stem = Path(path).stem.replace('_raw', ''); log(f'입력 {path} {src.size}')
    for i in range(a.n):
        s = a.seed + i; t0 = time.time()
        im = pipe(image=[src], prompt=P, negative_prompt=NEG, num_inference_steps=a.steps, true_cfg_scale=a.cfg,
                  height=a.size, width=a.size, generator=torch.Generator(device='cuda').manual_seed(s)).images[0]
        tag = f'{stem}' if a.n == 1 else f'{stem}_{s}'
        im.save(OUT / f'{tag}_raw.png'); apply_palette(im, pal).save(OUT / f'{tag}.png')
        log(f'{tag}: {time.time()-t0:.0f}s -> {OUT}/{tag}.png')
log('완료.')
