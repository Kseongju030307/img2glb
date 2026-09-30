#!/bin/bash
# sample_fan_2: front/back/right + left(=right 반전) 4뷰 -> GLB(octree1024, tex100만, GPU 3) -> 렌더. 옆모습 실험(sidefew)이 끝난 뒤 시작.
G=/data/sungju030307/img2glb/logs; V=/data/sungju030307/img2glb/generated/02_views_fan2; O=/data/sungju030307/img2glb/generated/03_glb/fan2

cd /home/sungju030307/img2glb && source env.sh && unset CUDA_VISIBLE_DEVICES
echo "[$(date +%H:%M:%S)] GLB 4뷰 시작 (GPU 1)"
python generate.py --input_dir $V --out $O --octree 1024 --flashvdm --paint_subfolder hunyuan3d-paint-v2-0 --paint_oblique --tex_faces 300000 --gpu 1 > $G/glb_fan2.log 2>&1 || { echo "GLB 실패 $?"; exit 1; }
python bench/render_tex.py $O/mesh_textured.glb $O/mesh_textured_views.png 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] 종료 코드 0"
