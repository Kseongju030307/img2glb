#!/usr/bin/env python
"""
생성 이미지를 '딱 N색 무광(flat)' 으로 강제한다. 디퓨전은 프롬프트로 색 수를 못 지키므로 후처리로 확정.

팔레트 모드
  fixed (권장): 색을 직접 지정. --colors "#f2d3c2,#ffffff,#8ec6ff,#f3a6c4,#000000" (피부·흰·하늘·분홍·검정)
  hue         : 색조 계열별로 1색씩 자동 추출 (--n)
  kmeans      : 순수 RGB k-means (밝기 단계도 색으로 셈)

매칭 규칙(fixed/hue): 픽셀을 '유채색'과 '무채색'으로 가른 뒤, 유채색은 팔레트의 유채색 중 **색조각**이 가장
가까운 것으로(밝기 무시 → 하늘색 3단계가 전부 하나의 하늘색), 무채색은 흰/검 중 밝기로.
유채/무채 경계(채도 문턱)는 이미지마다 다르므로 기준 이미지(front)에서 **Otsu 로 자동 보정**한다:
흰 후드(채도≈0~3) vs 하늘 파자마(≈8~30) 처럼 쌍봉 분포의 골짜기를 찾는다. 보정값은 모든 뷰에 공유.

  python palette.py --ref front.png --imgs front.png back.png left.png --colors "#..,#..,.." --out_dir out/
  (모듈) calibrate(front); apply_palette(img, palette)
"""
import argparse, os
import numpy as np
from PIL import Image

# ── 색 공간 ────────────────────────────────────────────────────────────────
def _rgb2lab(rgb):
    rgb = rgb.astype(np.float64) / 255.0
    m = rgb > 0.04045; rgb = np.where(m, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = rgb @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)

def _fg_mask(rgba, bg_thresh=235):
    a = np.asarray(rgba)
    if a.shape[-1] == 4 and (a[..., 3] < 255).any():
        return a[..., 3] > 0
    return a[..., :3].min(axis=-1) < bg_thresh

def _outline_mask(rgb, thresh=70):
    return rgb.max(axis=-1) < thresh

def _kmeans(X, n, seed=0, iters=40):
    rng = np.random.default_rng(seed)
    cent = [X[rng.integers(len(X))]]
    for _ in range(1, n):
        d = np.min(((X[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        cent.append(X[rng.choice(len(X), p=d / d.sum())])
    cent = np.array(cent)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        new = np.array([X[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(n)])
        if np.allclose(new, cent, atol=1e-3): break
        cent = new
    lab = np.argmin(((X[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    return cent, lab

def _otsu(x, lo=0.0, hi=40.0, bins=80):
    """1차원 값의 쌍봉 분포를 가르는 문턱. 골짜기가 없으면 None."""
    x = x[(x >= lo) & (x <= hi)]
    if len(x) < 500: return None
    h, e = np.histogram(x, bins=bins, range=(lo, hi)); h = h.astype(np.float64)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1e-9); m1 = (np.cumsum((h * c)[::-1])[::-1] / np.maximum(w1, 1e-9))
    var = w0[:-1] * w1[1:] * (m0[:-1] - m1[1:]) ** 2
    t = c[np.argmax(var)]
    # 양쪽에 실제로 덩어리가 있어야 의미 있는 문턱 (한쪽이 5% 미만이면 무시)
    if min((x < t).mean(), (x >= t).mean()) < 0.05: return None
    return float(t)

# ── 유채/무채 경계 ───────────────────────────────────────────────────────────
HUE_W = np.array([0.25, 1.0, 1.0])            # hue 모드 팔레트 추출용 가중치
THRESH = {'cool': 8.0, 'warm': 5.0, 'dark': 35.0}   # dark: L 이 이보다 어두우면 색조 무시하고 검정(눈동자·눈썹·선화)           # 기본 채도 문턱 (calibrate 로 이미지별 갱신)

def _is_cool(hdeg):
    return (hdeg > 180) & (hdeg < 300)

def calibrate(ref_img, keep_outline=True):
    """기준 이미지에서 파랑 계열/따뜻한 계열 각각 채도 분포의 골짜기(Otsu)를 찾아 THRESH 갱신."""
    rgba = ref_img.convert('RGBA'); a = np.asarray(rgba)
    m = _fg_mask(rgba)
    if keep_outline: m = m & ~_outline_mask(a[..., :3])
    L = _rgb2lab(a[..., :3][m].astype(np.float64))
    C = np.hypot(L[:, 1], L[:, 2]); h = np.degrees(np.arctan2(L[:, 2], L[:, 1])) % 360
    cool = _is_cool(h)
    for key, sel, default in (('cool', cool, 8.0), ('warm', ~cool, 5.0)):
        t = _otsu(C[sel]) if sel.any() else None
        THRESH[key] = float(np.clip(t, 3.0, 20.0)) if t is not None else default
    return dict(THRESH)

def _is_chromatic(C, hdeg):
    return np.where(_is_cool(hdeg), C >= THRESH['cool'], C >= THRESH['warm'])

def fit_palette(img, n=5, mode='hue', keep_outline=True, seed=0):
    rgba = img.convert('RGBA'); a = np.asarray(rgba)
    m = _fg_mask(rgba)
    if keep_outline: m = m & ~_outline_mask(a[..., :3])
    px = a[..., :3][m].astype(np.float64)
    if len(px) > 150000:
        px = px[np.random.default_rng(seed).choice(len(px), 150000, replace=False)]
    if mode == 'kmeans':
        cent, _ = _kmeans(px, n, seed); pal = cent
    else:
        _, lab = _kmeans(_rgb2lab(px) * HUE_W, n, seed)
        pal = np.array([np.median(px[lab == k], axis=0) if (lab == k).any() else px.mean(0) for k in range(n)])
    pal = pal[np.argsort(pal.sum(1))]
    return np.clip(pal, 0, 255).round().astype(np.uint8)

def parse_colors(s):
    out = []
    for t in s.split(','):
        t = t.strip().lstrip('#')
        if len(t) == 3: t = ''.join(c * 2 for c in t)
        out.append([int(t[i:i + 2], 16) for i in (0, 2, 4)])
    return np.array(out, np.uint8)

def _hue_match(src_rgb, pal_rgb):
    S = _rgb2lab(src_rgb); P = _rgb2lab(pal_rgb)
    cs = np.hypot(S[:, 1], S[:, 2]); cp = np.hypot(P[:, 1], P[:, 2])
    hs = np.degrees(np.arctan2(S[:, 2], S[:, 1])) % 360; hp = np.degrees(np.arctan2(P[:, 2], P[:, 1])) % 360
    pc = cp >= 5.0                                   # 팔레트 색 자체의 유채/무채 (하늘·피부·분홍 vs 흰·검)
    chrom = np.where(pc)[0]; achrom = np.where(~pc)[0]
    if len(chrom) == 0: chrom = np.arange(len(P))
    if len(achrom) == 0: achrom = np.arange(len(P))
    idx = np.empty(len(S), np.int64)
    is_c = _is_chromatic(cs, hs) & (S[:, 0] >= THRESH.get('dark', 0.0))
    dh = np.abs((np.radians(hs[is_c])[:, None] - np.radians(hp[chrom])[None] + np.pi) % (2 * np.pi) - np.pi)
    dc = np.abs(cs[is_c, None] - cp[None, chrom]) / 100.0
    idx[is_c] = chrom[np.argmin(dh + 0.3 * dc, axis=1)]
    dl = np.abs(S[~is_c, 0][:, None] - P[None, achrom, 0])
    idx[~is_c] = achrom[np.argmin(dl, axis=1)]
    return idx

def gradient_outline(img, sigma=1.0, thresh=7.0):
    """원본에 검은 선화가 없는(부드러운 셀 셰이딩) 그림용: Lab 공간 기울기(Sobel)가 큰 곳을 외곽선으로 잡는다.
    같은 하늘색 계열(머리/자켓)이 한 색으로 합쳐질 때 경계가 사라지는 문제를 막는다(2026-09-29 실측: 6~9 가 적당)."""
    from scipy import ndimage as ndi
    a = np.asarray(img.convert('RGB')).astype(np.float32)
    lab = _rgb2lab(a.reshape(-1, 3)).reshape(a.shape)
    g = np.zeros(a.shape[:2])
    for c in range(3):
        ch = ndi.gaussian_filter(lab[..., c], sigma)
        g += np.hypot(ndi.sobel(ch, 1), ndi.sobel(ch, 0))
    return ndi.binary_closing(g / 8.0 > thresh, iterations=1)

def apply_palette(img, palette, mode='hue', keep_outline=True, smooth=5, edge=None, white_bg=True, fg_all=False):
    """smooth>0: 픽셀 단위 매칭이 만드는 '도트' 잡음을 없앤다 — 매칭 전 중앙값 필터(색 경계의 안티앨리어싱
    혼색 제거) + 매칭 후 라벨맵 최빈값 필터(외딴 점 제거). 외곽선은 그대로."""
    from PIL import ImageFilter
    rgba = img.convert('RGBA'); a = np.asarray(rgba).copy()
    m = np.ones(a.shape[:2], bool) if fg_all else _fg_mask(rgba); rgb = a[..., :3]
    out = rgb.copy()
    src_img = Image.fromarray(rgb, 'RGB')
    if smooth > 0:
        src_img = src_img.filter(ImageFilter.MedianFilter(smooth))
    srgb = np.asarray(src_img)
    if keep_outline:
        # 외곽선은 (1) 필터된 이미지 기준으로, (2) 실루엣 가장자리(배경과 맞닿아 안티앨리어싱으로 어두워진 픽셀)는
        # 제외 — 전경 마스크를 조금 침식해서 그 안쪽에서만 검정을 인정. 안 그러면 테두리에 검은 점이 뿌려진다(실측).
        inner = np.asarray(Image.fromarray(m.astype(np.uint8) * 255, 'L').filter(ImageFilter.MinFilter(5))) > 0
        ol = inner & _outline_mask(srgb); out[ol] = 0; m = m & ~ol
    src = srgb[m].astype(np.float64); pal = palette.astype(np.float64)
    idx = np.empty(len(src), np.int64)
    for i in range(0, len(src), 200000):
        blk = src[i:i + 200000]
        if mode == 'kmeans':
            idx[i:i + 200000] = np.argmin(((blk[:, None, :] - pal[None]) ** 2).sum(-1), axis=1)
        else:
            idx[i:i + 200000] = _hue_match(blk, pal)
    if smooth > 0:
        # 라벨맵을 이미지로 만들어 최빈값 필터 → 고립된 점 제거. 배경은 별도 라벨(255) 로 두고 마스크 유지
        lab = np.full(m.shape, 255, np.uint8); lab[m] = idx
        lab = np.asarray(Image.fromarray(lab, 'L').filter(ImageFilter.ModeFilter(smooth)))
        lab = np.where(m, np.minimum(lab, len(palette) - 1), 255)
        idx = lab[m].astype(np.int64)
    out[m] = palette[idx]
    if edge:  # 기울기 외곽선 덧그리기 (전경 안쪽만)
        out[gradient_outline(img, 1.0, edge) & _fg_mask(rgba)] = 0
    if white_bg and not fg_all:  # 배경의 거의-흰색 잡색 제거 → 파일 전체가 정확히 팔레트 색만 갖게
        out[~_fg_mask(rgba)] = 255
    a[..., :3] = out
    return Image.fromarray(a, 'RGBA') if img.mode == 'RGBA' else Image.fromarray(a[..., :3], 'RGB')

def swatch(palette, size=64):
    im = Image.new('RGB', (size * len(palette), size))
    for i, c in enumerate(palette):
        im.paste(Image.new('RGB', (size, size), tuple(int(x) for x in c)), (i * size, 0))
    return im

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='팔레트/문턱을 뽑을 기준 이미지 (보통 front)')
    ap.add_argument('--imgs', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--mode', choices=['hue', 'kmeans'], default='hue')
    ap.add_argument('--colors', default=None, help='직접 지정 "#rrggbb,..." (있으면 --n/--mode 무시)')
    ap.add_argument('--no_outline', action='store_true')
    ap.add_argument('--calib', choices=['otsu', 'fixed'], default='fixed',
                    help='유채/무채 채도 문턱. fixed(기본)=고정값(--cool/--warm). 실측: 채도 높인 프롬프트로 생성한 '
                         '그림엔 고정 11/5 가 정확, otsu 는 피부를 흰색으로 보내는 등 불안정')
    ap.add_argument('--cool', type=float, default=11.0, help='fixed 일 때 파랑 계열 문턱')
    ap.add_argument('--warm', type=float, default=5.0, help='fixed 일 때 따뜻한 계열 문턱')
    ap.add_argument('--edge', type=float, default=None, help='Lab 기울기 외곽선 문턱(선화 없는 그림용, 6~9 권장). 없으면 안 그림')
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ref = Image.open(a.ref)
    if a.calib == 'otsu':
        th = calibrate(ref, keep_outline=not a.no_outline)
    else:
        THRESH.update(cool=a.cool, warm=a.warm); th = dict(THRESH)
    pal = parse_colors(a.colors) if a.colors else fit_palette(ref, a.n, a.mode, keep_outline=not a.no_outline)
    print('팔레트:', ['#%02x%02x%02x' % tuple(int(x) for x in c) for c in pal], '| 채도 문턱', {k: round(v, 1) for k, v in th.items()})
    swatch(pal).save(os.path.join(a.out_dir, 'palette.png'))
    for f in a.imgs:
        o = os.path.join(a.out_dir, os.path.basename(f))
        apply_palette(Image.open(f), pal, a.mode, keep_outline=not a.no_outline, edge=a.edge).save(o); print(' ->', o)
