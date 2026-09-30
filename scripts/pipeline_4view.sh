#!/bin/bash
# 옆모습(768, 치비 비례 + 앉히기 재시도) -> 정렬/반전 4뷰 -> GLB(octree1024, tex100만) -> 텍스처 5색 양자화 -> 렌더
G=/data/sungju030307/img2glb/logs; D=/data/sungju030307/img2glb/generated/02_turnaround_768; O=/data/sungju030307/img2glb/generated/03_glb/s4_768_4view
cd /home/sungju030307/img2glb && source env.sh && unset CUDA_VISIBLE_DEVICES; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] 1/4 옆모습 768"
python turnaround.py --front inputs/smaple4/front.png --gpus 1,2,3 --views front,left --steps 12 --cfg 3.0 --size 768 --out $D/side2 > $G/turn_768_side2.log 2>&1 || { echo "옆모습 실패 $?"; exit 1; }
echo "[$(date +%H:%M:%S)] 2/4 정렬/반전"
python align_views.py --front $D/front.png --back $D/back.png --left $D/side2/left.png --out $D/four || exit 1
echo "[$(date +%H:%M:%S)] 3/4 GLB 4뷰 (GPU 3)"
python generate.py --input_dir $D/four --out $O --octree 1024 --flashvdm --paint_subfolder hunyuan3d-paint-v2-0 --paint_oblique --tex_faces 300000 --gpu 3 > $G/glb_4view.log 2>&1 || { echo "GLB 실패 $?"; exit 1; }
echo "[$(date +%H:%M:%S)] 4/4 5색 양자화 + 렌더"
python quantize_glb.py $O/mesh_textured.glb $O/mesh_5color.glb --render 2>&1 | grep -v Warn
python bench/render_tex.py $O/mesh_textured.glb $O/mesh_textured_views.png 2>&1 | tail -1
echo "[$(date +%H:%M:%S)] 종료 코드 0"
