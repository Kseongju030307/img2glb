"""옆모습 실험: 앉은 정면+뒷모습 2장을 few-shot 참조로, '3D 프린팅용 정사영 턴어라운드' 맥락으로 옆모습 요청(사용자 제안 2026-09-30)."""
import os, sys, time, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'; os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
from pathlib import Path
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
D = Path('/data/sungju030307/img2glb/generated/02_turnaround_768'); OUT = D / 'side_fewshot'; OUT.mkdir(exist_ok=True)
def log(*m): print(time.strftime('[%H:%M:%S]'), *m, flush=True)
pipe = QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.bfloat16, device_map='balanced')
F = Image.open(D / 'front.png').convert('RGB'); B = Image.open(D / 'back.png').convert('RGB')
CTX = ('Picture 1 is the front view and Picture 2 is the back view of the same chibi character, drawn as an orthographic '
       'character turnaround sheet for 3D printing modeling: same scale, same pose, same proportions in every view. ')
SIDE = ('Draw the missing LEFT SIDE view of this turnaround: a strict 90 degree profile, the face in profile with only one eye '
        'visible, the nose pointing to the left edge, the back of the head on the right. ')
TAIL = ('Same character, same hairstyle, same hood, same outfit and colors, same head size and same height in frame as Picture 1. '
        'Plain white background, anime cel shading, no shadow, full body centered.')
NEG = ('multiple characters, two figures, side-by-side panels, grid, collage, text, watermark, cropped body, different character, '
       'different outfit, cast shadow, floor, perspective distortion, close-up, front view, face turned to camera, both eyes visible')
V = {
 'a_refpose':  CTX + SIDE + 'Keep exactly the same pose as in Picture 1 and Picture 2. ' + TAIL,
 'b_seated':   CTX + SIDE + 'The character is seated on the ground with legs folded, exactly as in Picture 1 and Picture 2, so the body is a low compact shape. ' + TAIL,
 'c_refpose_cfg4': CTX + SIDE + 'Keep exactly the same pose as in Picture 1 and Picture 2. ' + TAIL,
}
for name, p in V.items():
    cfg = 4.0 if 'cfg4' in name else 3.0; t0 = time.time()
    im = pipe(image=[F, B], prompt=p, negative_prompt=NEG, num_inference_steps=12, true_cfg_scale=cfg, height=768, width=768,
              generator=torch.Generator(device='cuda').manual_seed(0)).images[0]
    im.save(OUT / f'{name}.png'); log(f'{name}: {time.time()-t0:.0f}s')
S = 384; W = Image.new('RGB', (S * (2 + len(V)), S), 'white'); W.paste(F.resize((S, S)), (0, 0)); W.paste(B.resize((S, S)), (S, 0))
for i, n in enumerate(V): W.paste(Image.open(OUT / f'{n}.png').resize((S, S)), ((2 + i) * S, 0))
W.save(OUT / 'sheet.png'); log('완료.')
