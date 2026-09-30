"""앞+뒤 그림을 few-shot 참조로 주고 디퓨전으로 옆모습 생성 (fan2 등 '있는 그대로' 캐릭터용, 앉기/치비 지시 없음).
  python bench/side_fewshot_gen.py --front f.png --back b.png --out dir [--gpus 2,3] [--seeds 0 1]"""
import argparse, os, sys, time
ap = argparse.ArgumentParser(); ap.add_argument('--front', required=True); ap.add_argument('--back', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--gpus', default='2,3'); ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1]); ap.add_argument('--size', type=int, default=768)
ap.add_argument('--desc', default='', help='캐릭터 설명 추가(선택)')
a = ap.parse_args(); os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus; os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
import torch; from pathlib import Path; from PIL import Image; from diffusers import QwenImageEditPlusPipeline
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
def log(*m): print(time.strftime('[%H:%M:%S]'), *m, flush=True)
pipe = QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509', torch_dtype=torch.bfloat16, device_map='balanced')
F = Image.open(a.front).convert('RGB'); B = Image.open(a.back).convert('RGB')
P = ('Picture 1 is the front view and Picture 2 is the back view of the same character, drawn as an orthographic character turnaround '
     'sheet for 3D printing modeling: same scale, same pose, same proportions in every view. '
     'Draw the missing LEFT SIDE view of this turnaround: a strict 90 degree profile, the face in profile, the nose pointing to the left edge, '
     'the back of the head on the right. Keep exactly the same pose, the same height in frame and the same shapes as in Picture 1 and Picture 2. '
     f'Same character, same design, same colors{(", " + a.desc) if a.desc else ""}. Plain white background, soft clean rendering like the reference, no shadow, centered.')
NEG = ('multiple characters, two figures, side-by-side panels, grid, collage, text, watermark, cropped, different character, '
       'cast shadow, floor, perspective distortion, close-up, front view, face turned to camera, both eyes visible')
for s in a.seeds:
    t0 = time.time()
    im = pipe(image=[F, B], prompt=P, negative_prompt=NEG, num_inference_steps=12, true_cfg_scale=3.0, height=a.size, width=a.size,
              generator=torch.Generator(device='cuda').manual_seed(s)).images[0]
    im.save(OUT / f'left_seed{s}.png'); log(f'seed {s}: {time.time()-t0:.0f}s -> {OUT}/left_seed{s}.png')
S = 384; W = Image.new('RGB', (S * (2 + len(a.seeds)), S), 'white'); W.paste(F.resize((S, S)), (0, 0)); W.paste(B.resize((S, S)), (S, 0))
for i, s in enumerate(a.seeds): W.paste(Image.open(OUT / f'left_seed{s}.png').resize((S, S)), ((2 + i) * S, 0))
W.save(OUT / 'sheet.png'); log('완료.')
