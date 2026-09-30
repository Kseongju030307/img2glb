#!/usr/bin/env python
"""턴어라운드 뷰 정렬: 옆모습(left)을 정면 바운딩박스의 세로 크기/위치에 맞춰 스케일·이동하고, right 는 left 좌우 반전.
4뷰 3D(front/back/left/right)에서 옆모습 크기가 어긋나 형상이 나빠지던 문제 보정(2026-09-29).

  python align_views.py --front f.png --back b.png --left l.png --out dir
"""
import argparse, numpy as np
from PIL import Image, ImageOps
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--front', required=True); ap.add_argument('--back', required=True); ap.add_argument('--left', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--bg_thresh', type=int, default=235)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
def bbox(im):
    g = np.asarray(im.convert('RGB')).min(-1) < a.bg_thresh
    ys, xs = np.where(g); return ys.min(), ys.max(), xs.min(), xs.max()
def fit_to(im, ref):
    """im 의 전경 세로 범위를 ref 의 세로 범위와 같게 스케일하고, 세로 중심/가로 중심을 맞춘다."""
    y0, y1, x0, x1 = bbox(im); Y0, Y1, X0, X1 = bbox(ref)
    s = (Y1 - Y0) / max(1, y1 - y0)
    crop = im.crop((x0, y0, x1 + 1, y1 + 1)).resize((max(1, round((x1 - x0 + 1) * s)), Y1 - Y0 + 1), Image.LANCZOS)
    canvas = Image.new('RGB', ref.size, 'white')
    cx = (X0 + X1) // 2 - crop.width // 2
    canvas.paste(crop, (cx, Y0)); return canvas, s
F = Image.open(a.front).convert('RGB'); B = Image.open(a.back).convert('RGB'); L = Image.open(a.left).convert('RGB')
L2, s = fit_to(L, F); print(f'left 스케일 x{s:.3f}')
R2 = ImageOps.mirror(L2)
F.save(out / 'front.png'); B.save(out / 'back.png'); L2.save(out / 'left.png'); R2.save(out / 'right.png')
W = Image.new('RGB', (F.width * 4, F.height), 'white')
for i, im in enumerate([F, L2, B, R2]): W.paste(im, (i * F.width, 0))
W.resize((W.width // 2, W.height // 2)).save(out / 'sheet.png'); print('->', out)
