#!/usr/bin/env python
"""
뒷면 입력에 리본 고리가 안 그려져 있으면 리본 고리의 뒷면이 검게 칠해짐.
빨간 영역 근처(3D 바운딩 박스 안)의 어두운 면을 리본 색으로 다시 칠하는 후처리.
사용: python fix_bow.py outputs/sample1/mesh_textured.glb   -> 같은 폴더에 mesh_textured_fixed.glb
"""
import sys, numpy as np, trimesh
from PIL import Image
src = sys.argv[1]
m = trimesh.load(src, force='mesh')
tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)/255
uv = m.visual.uv; V, F = m.vertices, m.faces
H, W = tex.shape[:2]
fuv = uv[F].mean(1); px = np.clip((fuv[:,0]*(W-1)).astype(int),0,W-1); py = np.clip(((1-fuv[:,1])*(H-1)).astype(int),0,H-1)
col = tex[py, px]
red = (col[:,0] > 0.45) & (col[:,1] < 0.35) & (col[:,2] < 0.35)
dark = col.max(1) < 0.30
cen = V[F].mean(1)
lo, hi = cen[red].min(0) - 0.03, cen[red].max(0) + 0.03
inbox = np.all((cen >= lo) & (cen <= hi), 1)
target = dark & inbox
bow_rgb = np.median(col[red], 0) * 0.9         # 리본보다 살짝 어두운 빨강 (안쪽 면)
print(f'red faces {red.sum()}, dark-in-bow-box faces {target.sum()} -> recolor to {np.round(bow_rgb*255).astype(int)}')
# 해당 면들의 UV 삼각형을 텍스처에 직접 칠함
from PIL import ImageDraw
img = m.visual.material.baseColorTexture.convert('RGB'); d = ImageDraw.Draw(img)
c = tuple(int(x*255) for x in bow_rgb)
for f in np.where(target)[0]:
    tri = [(u*(W-1), (1-v)*(H-1)) for u, v in uv[F[f]]]
    d.polygon(tri, fill=c, outline=c)
m.visual.material.baseColorTexture = img
out = src.replace('mesh_textured.glb', 'mesh_textured_fixed.glb')
m.export(out); print('saved', out)
