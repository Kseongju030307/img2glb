# 실험: "옆모습 + 앉은 자세" 를 한 번에 요구하면 정면으로 되돌아간다(3회 실측).
# 두 단계로 쪼갠다: (1) 옆모습만 (이미 있음: turn_s4_cfg3/left.png, 서 있는 옆모습)
#                    (2) 그 이미지를 참조로 "포즈만 앉은 자세로" 편집. 정면(앉은)도 참조로 같이.
import os, time; os.environ.setdefault('HF_HUB_CACHE','/data/sungju030307/hf-cache')
import torch; from PIL import Image
from diffusers import QwenImageEditPlusPipeline
def log(*m): print(time.strftime('[%H:%M:%S]'),*m,flush=True)
pipe=QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509',torch_dtype=torch.bfloat16,device_map='balanced')
side=Image.open('/data/sungju030307/img2glb/outputs/turn_s4_cfg3/left.png').convert('RGB')
front=Image.open('/data/sungju030307/img2glb/outputs/turn_s4_v3/front.png').convert('RGB')
P=('Picture 1 is the character standing, seen from the left side in profile. Picture 2 is the same character '
   'sitting on the ground with legs folded, seen from the front. Redraw Picture 1 keeping its exact left side '
   'profile view and design, but change only the pose to the sitting pose of Picture 2: sitting on the ground, '
   'legs folded in front, compact body, same size in frame as Picture 2. Plain white background, anime cel shading.')
for cfg in (3.0, 4.5):
    t0=time.time()
    im=pipe(image=[side,front],prompt=P,num_inference_steps=12,true_cfg_scale=cfg,height=512,width=512,
            generator=torch.Generator(device='cuda').manual_seed(0)).images[0]
    p=f'/data/sungju030307/img2glb/outputs/turn_s4_v3/left_refined_cfg{cfg}.png'; im.save(p)
    log(f'cfg {cfg}: {time.time()-t0:.0f}s -> {p}')
log('완료.')
