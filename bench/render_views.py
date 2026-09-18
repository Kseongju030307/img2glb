#!/usr/bin/env python
"""
메시를 정해진 방위각에서 직교 투영으로 렌더링 (학습 데이터 생성 및 뷰 규칙 검증용).
회전 정의: 물체를 Y축 기준 +theta 도 회전시킨 뒤 +Z 에서 바라봄.
"""
import numpy as np, trimesh
from PIL import Image

def rotY(P, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return P @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T

def render_ortho(mesh, azim_deg, size=512, border=0.12, light=(0.3, 0.45, 0.85), bg=None):
    """RGBA 반환. bg=None 이면 투명 배경."""
    V = rotY(np.asarray(mesh.vertices, np.float64), azim_deg)
    N = rotY(np.asarray(mesh.face_normals, np.float64), azim_deg)
    F = mesh.faces
    c = (V.min(0) + V.max(0)) / 2
    V = V - c
    scale = np.abs(V[:, :2]).max()
    V[:, :2] /= scale * (1 + border)                       # -1~1 에 맞춤, 여백 확보
    l = np.array(light, np.float64); l /= np.linalg.norm(l)
    shade = np.clip(np.abs(N @ l), 0.0, 1.0) * 0.75 + 0.25

    tri = V[F]
    depth = tri[:, :, 2].mean(1)
    order = np.argsort(depth)                              # 뒤 -> 앞
    px = ((tri[..., 0] * 0.5 + 0.5) * (size - 1))
    py = ((-tri[..., 1] * 0.5 + 0.5) * (size - 1))

    # z-버퍼 없이 painter's algorithm 로 삼각형 스캔라인 채우기
    buf = np.zeros((size, size), np.float32)
    mask = np.zeros((size, size), bool)
    for i in order:
        x = px[i]; y = py[i]
        x0, x1 = int(np.floor(x.min())), int(np.ceil(x.max()))
        y0, y1 = int(np.floor(y.min())), int(np.ceil(y.max()))
        x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, size - 1); y1 = min(y1, size - 1)
        if x1 < x0 or y1 < y0: continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = ((y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2]))
        if abs(d) < 1e-12: continue
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w0 + w1 <= 1 + 1e-6)
        if not inside.any(): continue
        buf[gy[inside], gx[inside]] = shade[i]
        mask[gy[inside], gx[inside]] = True
    g = (np.clip(buf, 0, 1) * 255).astype(np.uint8)
    rgb = np.dstack([g, g, g])
    if bg is not None:
        rgb[~mask] = bg
        a = np.full((size, size), 255, np.uint8)
    else:
        rgb[~mask] = 255
        a = (mask * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, a]), 'RGBA')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh'); ap.add_argument('outdir')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--faces', type=int, default=200000, help='렌더 전 면 수 축소 (0=안함)')
    a = ap.parse_args()
    import os; os.makedirs(a.outdir, exist_ok=True)
    m = trimesh.load(a.mesh, force='mesh')
    if a.faces and len(m.faces) > a.faces:
        m = m.simplify_quadric_decimation(face_count=a.faces)
    for az in (0, 90, 180, 270):
        render_ortho(m, az, a.size).save(f'{a.outdir}/az{az:03d}.png')
        print(f'  az {az:3d} -> {a.outdir}/az{az:03d}.png')
