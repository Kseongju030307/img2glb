#!/usr/bin/env python
"""
배경 제거. 단색 배경(흰색 등)이면 테두리에서 flood-fill 로 딱 떨어지게 키잉하고,
아니면 rembg 로 폴백. rembg 는 검정 머리 같은 큰 어두운 영역을 배경으로 오인하거나
알파를 반투명하게 만들어서 플랫 컬러 캐릭터에는 잘 안 맞음.

사용: python bgremove.py in.png out.png [--mode auto|key|rembg]
"""
import argparse
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

def border_color(rgb):
    b = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3), rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    return b.mean(0), b.std(0).max()

def key_background(img: Image.Image, tol=14, feather=1.0):
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    bg, _ = border_color(rgb)
    close = np.abs(rgb - bg).max(-1) <= tol                 # 배경색과 비슷한 픽셀
    # 테두리에서 연결된 영역만 배경으로 (캐릭터 안쪽의 흰 부분은 보존)
    lbl, n = ndi.label(close)
    border_labels = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
    bgmask = np.isin(lbl, border_labels[border_labels > 0])
    alpha = (~bgmask).astype(np.float32)
    if feather > 0:
        alpha = ndi.gaussian_filter(alpha, feather)
        alpha = np.clip((alpha - 0.3) / 0.4, 0, 1)          # 살짝만 부드럽게, 대부분 0/1
    out = np.concatenate([rgb, alpha[..., None] * 255], -1).astype(np.uint8)
    return Image.fromarray(out, "RGBA")

def remove_background(img: Image.Image, mode="auto", tol=14):
    """RGBA 반환. mode: auto(단색이면 key, 아니면 rembg) | key | rembg"""
    if img.mode == "RGBA" and mode == "auto":
        a = np.asarray(img)[..., 3]
        if (a < 255).mean() > 0.01:          # 진짜 투명 배경이면 그대로
            return img
        # 알파가 전부 불투명인 RGBA = 사실상 RGB. 배경 제거 필요
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    _, spread = border_color(rgb)
    if mode == "key" or (mode == "auto" and spread < 6):
        return key_background(img, tol=tol)
    from hy3dgen.rembg import BackgroundRemover
    return BackgroundRemover()(img.convert("RGB"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--mode", default="auto", choices=["auto", "key", "rembg"]); ap.add_argument("--tol", type=int, default=14, help="배경색 허용 오차. 연한 그림자까지 지우려면 30~50")
    a = ap.parse_args()
    out = remove_background(Image.open(a.src), a.mode, a.tol)
    out.save(a.dst); al = np.asarray(out)[..., 3]
    print(f"{a.dst}: 불투명 {np.mean(al == 255):.2f}, 투명 {np.mean(al == 0):.2f}")
