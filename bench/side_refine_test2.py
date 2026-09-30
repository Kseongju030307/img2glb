# 실험 2: 앉은 정면을 참조로 주면 그걸 복제해 버린다(cfg 3/4.5 둘 다). 정면 참조를 빼고
# 서 있는 옆모습 한 장만 넣어, "sitting" 단어 없이 자세를 기하학적으로 서술한다.
import os, time; os.environ.setdefault('HF_HUB_CACHE','/data/sungju030307/hf-cache')
import torch; from PIL import Image
from diffusers import QwenImageEditPlusPipeline
def log(*m): print(time.strftime('[%H:%M:%S]'),*m,flush=True)
pipe=QwenImageEditPlusPipeline.from_pretrained('Qwen/Qwen-Image-Edit-2509',torch_dtype=torch.bfloat16,device_map='balanced')
side=Image.open('/data/sungju030307/img2glb/outputs/turn_s4_cfg3/left.png').convert('RGB')
PROMPTS={
 'geo': ('Keep this exact left side profile view of the character and the same design. Change only the legs: '
         'bend the knees and fold the legs so the character rests on the ground, the body becomes a low compact '
         'shape, the head stays at the top. Plain white background, anime cel shading.'),
 'kneel': ('Same character, same left side profile view, same outfit. Now the character is kneeling down on the '
           'ground with knees bent under the body, shown from the side. Plain white background, anime cel shading.'),
}
for tag,P in PROMPTS.items():
    t0=time.time()
    im=pipe(image=[side],prompt=P,num_inference_steps=12,true_cfg_scale=3.5,height=512,width=512,
            generator=torch.Generator(device='cuda').manual_seed(0)).images[0]
    p=f'/data/sungju030307/img2glb/outputs/turn_s4_v3/left_refined2_{tag}.png'; im.save(p)
    log(f'{tag}: {time.time()-t0:.0f}s -> {p}')
log('완료.')
