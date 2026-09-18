#!/usr/bin/env python
"""
플랫 컬러 2D 그림에 가짜 음영을 넣어 3D 모델이 '평면 스티커'로 오해하지 않게 하는 전처리.
실루엣(알파)에서 거리장을 만들어 부풀린 높이맵 → 법선 → 좌상단 조명으로 셰이딩 + 가장자리 AO.

사용: python shade.py inputs/sample1 inputs/sample1_shaded [--strength 0.45] [--ao 0.25]
"""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

def shade_image(img: Image.Image, strength=0.45, ao=0.25, puff=0.5, light=(-0.5, -0.6, 0.65)):
    rgba = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
    rgb, a = rgba[..., :3], rgba[..., 3]
    mask = a > 0.5
    if mask.sum() == 0:
        return img
    # 안쪽 거리장 → 부풀린 높이 (sqrt 로 둥글게), 크기 정규화
    dist = ndi.distance_transform_edt(mask)
    h = np.sqrt(dist / max(dist.max(), 1.0))
    h = ndi.gaussian_filter(h, sigma=max(2, min(mask.shape) / 200)) * puff
    gy, gx = np.gradient(h)
    n = np.stack([-gx, -gy, np.ones_like(h) * 0.02], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    l = np.array(light, np.float32); l /= np.linalg.norm(l)
    lambert = np.clip(n @ l, 0, 1)
    # 조명: 중앙 밝고 가장자리로 갈수록 어두워짐
    shade = (1 - strength) + strength * lambert
    # AO: 실루엣 가장자리 살짝 어둡게
    edge = np.clip(dist / (0.06 * max(mask.shape)), 0, 1)
    shade *= (1 - ao) + ao * edge
    out = rgb * shade[..., None]
    # 어두운 선(외곽선)은 그대로 두고, 검정 영역엔 약한 하이라이트를 줘서 형태가 읽히게
    dark = rgb.mean(-1) < 0.15
    out[dark] = rgb[dark] + (lambert[dark, None] * 0.18)
    out = np.clip(out, 0, 1)
    res = np.concatenate([out, a[..., None]], -1)
    return Image.fromarray((res * 255).astype(np.uint8), "RGBA")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--strength", type=float, default=0.45, help="음영 세기 0~1")
    ap.add_argument("--ao", type=float, default=0.25, help="가장자리 어둡게 0~1")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        im = Image.open(f)
        if im.mode != "RGBA":
            print(f"[skip] {f.name}: 투명 배경(RGBA) 아님"); continue
        shade_image(im, a.strength, a.ao).save(dst / (f.stem + ".png"))
        print(f"[ok] {f.name} -> {dst / (f.stem + '.png')}")

if __name__ == "__main__":
    main()
