# 실험 3: 옆모습 2단계에서 두 번째 참조로 '뒷모습'(앉은 포즈·크기·의상 정보는 있고 얼굴은 없음)을 준다.
# 앉은 정면을 주면 복제해 버리고(3회), 옆모습 한 장만 주면 비례가 어긋난다(1회) — 그 사이를 노린다.
import os, time; os.environ.setdefault('HF_HUB_CACHE','/data/sungju030307/hf-cache')
import torch; from PIL import Image
from diffusers import QwenImageEditPlusPipeline
def log(*m): print(time.strftime('[%H:%M:%S]'),*m,flush=True)
pipe=QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509',torch_dtype=torch.bfloat16,device_map='balanced')
O='/data/sungju030307/img2glb/generated/02_turnaround/'
side=Image.open(O+'experiments/left_standing.png').convert('RGB')      # 서 있는 옆모습
back=Image.open(O+'back.png').convert('RGB')                        # 앉은 뒷모습
P=('Picture 1 is the character standing, seen from the left side in profile. Picture 2 is the same character '
   'seen from behind, resting low on the ground. Redraw Picture 1 keeping its exact left side profile view, face '
   'and design, but change only the legs: bend the knees and fold the legs so the character rests on the ground '
   'like in Picture 2, the body becomes a low compact shape with the same height and size in frame as Picture 2. '
   'Plain white background, anime cel shading.')
for cfg,steps in ((3.5,20),):
    t0=time.time()
    im=pipe(image=[side,back],prompt=P,num_inference_steps=steps,true_cfg_scale=cfg,height=512,width=512,
            generator=torch.Generator(device='cuda').manual_seed(0)).images[0]
    p=f'{O}experiments/left_refined3_back_s{steps}.png'; im.save(p); log(f'cfg {cfg} steps {steps}: {time.time()-t0:.0f}s -> {p}')
log('완료.')
