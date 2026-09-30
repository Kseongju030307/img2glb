#!/bin/bash
# tabi.stl 에 sample2 뷰로 텍스처만 입혀 GLB 로. /data 가 안 붙어 있으면 루트 디스크(여유 1.7GB)에
# 쓰게 되므로 반드시 마운트 확인 후 실행.
set -e
if ! mountpoint -q /data; then echo "/data 가 마운트되지 않음. 먼저: sudo mount -a"; exit 1; fi
cd /home/sungju030307/img2glb; source env.sh; unset CUDA_VISIBLE_DEVICES
I=/data/sungju030307/img2glb/inputs_rgba/smaple2
if [ ! -f $I/front.png ]; then
  python3 -c "
import sys; sys.path.insert(0,'.')
from bgremove import remove_background
from PIL import Image; import glob,os
d='$I'; os.makedirs(d,exist_ok=True)
for f in sorted(glob.glob('inputs/smaple2/*.png')): remove_background(Image.open(f),'auto').save(d+'/'+f.split('/')[-1])
print('배경 제거 완료')"
fi
YAW=${YAW:-0}
python generate.py --mesh inputs/stl_tabi/tabi.stl --mesh_normalize --mesh_up z --mesh_yaw $YAW \
  --input_dir $I --gpu ${GPU:-0} --tex_faces 1000000 \
  --paint_subfolder hunyuan3d-paint-v2-0 --paint_oblique \
  --out /home/sungju030307/img2glb/outputs/tabi_tex${YAW:+_yaw$YAW}
