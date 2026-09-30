#!/usr/bin/env python
"""GLB 의 베이스컬러 텍스처만 고정 N색(무광)으로 양자화한다. 형상/UV 는 그대로.
일반 텍스처(원본 턴어라운드로 구운 것)를 뽑은 뒤 텍스처 이미지에만 팔레트를 씌우는 방식 —
2D 이미지를 미리 5색으로 만들면 하늘색끼리 경계가 사라져 이상해지므로(2026-09-29) 이 순서가 낫다.

  python quantize_glb.py in.glb out.glb [--colors "#..,#.."] [--cool 5 --warm 4] [--unlit] [--render]
"""
import argparse, sys
sys.path.insert(0, '/home/sungju030307/img2glb')
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--colors', default='#f2d3c2,#ffffff,#8ec6ff,#f3a6c4,#000000')
ap.add_argument('--cool', type=float, default=5.0, help='파랑 계열 채도 문턱. 구운 텍스처는 채도가 옅어져 2D(11)보다 낮게')
ap.add_argument('--warm', type=float, default=4.0)
ap.add_argument('--smooth', type=int, default=5)
ap.add_argument('--unlit', action='store_true', help='KHR_materials_unlit — 뷰어 조명 없이 완전 무광 평면색')
ap.add_argument('--render', action='store_true', help='4방향 렌더 sheet 저장 (dst 옆 *_views.png)')
a = ap.parse_args()
import numpy as np, trimesh
from PIL import Image
from palette import apply_palette, parse_colors, THRESH
from generate import fix_glb_material
THRESH.update(cool=a.cool, warm=a.warm); pal = parse_colors(a.colors)
m = trimesh.load(a.src, force='mesh')
tex = m.visual.material.baseColorTexture.convert('RGB')
# 텍스처 아틀라스는 배경이 없으므로(빈 UV 영역도 채워짐) 전경 마스크 없이 전체를 매칭
q = apply_palette(tex, pal, keep_outline=True, smooth=a.smooth, white_bg=False, fg_all=True)
u = np.unique(np.asarray(q).reshape(-1, 3), axis=0); print(f'텍스처 {tex.size} -> {len(u)}색')
m.visual.material.baseColorTexture = q
m.export(a.dst); fix_glb_material(a.dst, a.unlit); print('->', a.dst)
if a.render:
    from bench.render_tex import render_sheet
    render_sheet(a.dst, a.dst.replace('.glb', '_views.png'))
