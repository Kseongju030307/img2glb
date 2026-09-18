#!/usr/bin/env python
"""
GT 메시 -> 4뷰 렌더 + GT latent + DINO 조건 임베딩. 품질 게이트 포함.
GPU 한 장당 한 프로세스로 돌리고, 데이터를 shard 로 나눠 병렬 처리한다.

  python train/preprocess.py --n 300 --nshards 4 --shard 0 --gpu 0

출력: <cache>/<uid>/{latent.npy, cond.npy, views/*.png, meta.json}
이미 있으면 건너뛰므로 중단 후 재실행해도 안전하다.
"""
import argparse, os, sys, json, time, gc, traceback
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--list', default='/data/sungju030307/objaverse_meta/hiprec_anime.json')
ap.add_argument('--cache', default='/data/sungju030307/img2glb/train/cache')
ap.add_argument('--n', type=int, default=300, help='목록 앞에서 몇 개까지 시도할지')
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--gpu', type=int, default=0)
ap.add_argument('--size', type=int, default=512)
ap.add_argument('--chamfer_max', type=float, default=0.025, help='VAE 재현 오차 상한')
ap.add_argument('--min_faces', type=int, default=20000)
ap.add_argument('--max_faces', type=int, default=200000)
ap.add_argument('--octree_check', type=int, default=256, help='품질 게이트용 디코드 해상도')
a = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
os.environ.setdefault('HF_HUB_CACHE', '/data/sungju030307/hf-cache')
# objaverse 는 기본으로 ~/.objaverse 에 받는다. 홈(루트) 디스크는 여유가 거의 없으므로
# /data 로 돌린다. 안 그러면 1229개(약 30GB) 받다가 루트가 가득 찬다. (실제로 겪음)
os.environ.setdefault('OBJAVERSE_HOME', '/data/sungju030307/objaverse_cache')
import objaverse as _ov
_ov.BASE_PATH = '/data/sungju030307/objaverse_cache'
_ov._VERSIONED_PATH = os.path.join(_ov.BASE_PATH, 'hf-objaverse-v1')
os.makedirs(_ov._VERSIONED_PATH, exist_ok=True)
REPO = Path('/data/sungju030307/img2glb/Hunyuan3D-2')
sys.path.insert(0, str(REPO)); sys.path.insert(0, '/home/sungju030307/img2glb')

import numpy as np, torch, trimesh, objaverse
from scipy.spatial import cKDTree
from bench.render_views import render_ortho
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen.models.autoencoders import ShapeVAE
from hy3dgen.shapegen.surface_loaders import SharpEdgeSurfaceLoader

DEV = 'cuda:0'
VIEW_AZIM = {'front': 0, 'left': 90, 'back': 180, 'right': 270}
CACHE = Path(a.cache); CACHE.mkdir(parents=True, exist_ok=True)
TAG = f'[shard {a.shard}/{a.nshards} gpu{a.gpu}]'

def log(*m): print(TAG, *m, flush=True)

log('모델 로드 중')
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv', subfolder='hunyuan3d-dit-v2-mv', variant='fp16', device=DEV)
DT = next(pipe.model.parameters()).dtype
vae = ShapeVAE.from_pretrained('tencent/Hunyuan3D-2', subfolder='hunyuan3d-vae-v2-0-withencoder',
                               use_safetensors=False, pc_size=30720, pc_sharpedge_size=30720).to(DEV).eval()
loader = SharpEdgeSurfaceLoader(num_sharp_points=30720, num_uniform_points=30720)

def pts(m, n=30000):
    v, _ = trimesh.sample.sample_surface(m, n)
    v = np.asarray(v) - m.bounds.mean(0)
    return v / np.abs(v).max()

@torch.no_grad()
def cond_of(imgs):
    ci = pipe.prepare_image(imgs)                      # 반드시 prepare_image 경유
    img = ci.pop('image').to(DEV, dtype=DT)
    return pipe.encode_cond(image=img, additional_cond_inputs=ci,
                            do_classifier_free_guidance=False, dual_guidance=False)['main']

recs = json.load(open(a.list))[:a.n]
mine = recs[a.shard::a.nshards]
log(f'담당 {len(mine)}개 / 전체 {len(recs)}개')

stat = {'ok': 0, 'skip': 0, 'few_faces': 0, 'vae_fail': 0, 'error': 0}
t0 = time.time()
for i, r in enumerate(mine):
    d = CACHE / r['uid']
    if (d/'latent.npy').exists() and (d/'cond.npy').exists():
        stat['skip'] += 1; continue
    try:
        p = objaverse.load_objects(uids=[r['uid']], download_processes=1)[r['uid']]
        sc = trimesh.load(p, process=False)
        m = sc.to_geometry() if hasattr(sc, 'to_geometry') else sc
        del sc; gc.collect()
        if m.is_empty or len(m.faces) < a.min_faces:
            stat['few_faces'] += 1; continue
        if len(m.faces) > a.max_faces:
            m = m.simplify_quadric_decimation(face_count=a.max_faces)
        m.apply_translation(-m.bounds.mean(0))
        m.apply_scale(1.0 / np.abs(m.vertices).max())
        d.mkdir(parents=True, exist_ok=True)
        g = d/'mesh.glb'; m.export(g)

        with torch.no_grad():
            lat = vae.encode(loader(str(g)).to(DEV, dtype=torch.float16))
            o = vae.latents2mesh(vae.decode(lat), output_type='trimesh', bounds=1.01, mc_level=0.0,
                                 num_chunks=20000, octree_resolution=a.octree_check,
                                 mc_algo='mc', enable_pbar=False)[0]
        rt = trimesh.Trimesh(vertices=o.mesh_v, faces=o.mesh_f)
        if rt.is_empty or len(rt.faces) < 100: raise RuntimeError('빈 복원')
        A, B = pts(m), pts(rt)
        err = float((cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean()) / 2)
        if err > a.chamfer_max:
            stat['vae_fail'] += 1
            for f in d.glob('*'): f.unlink()
            d.rmdir(); continue

        rr = m.simplify_quadric_decimation(face_count=120000) if len(m.faces) > 120000 else m
        (d/'views').mkdir(exist_ok=True)
        imgs = {}
        for k, az in VIEW_AZIM.items():
            im = render_ortho(rr, az, a.size)
            im.save(d/'views'/f'{k}.png'); imgs[k] = im
        np.save(d/'latent.npy', lat[0].float().cpu().numpy().astype(np.float16))
        np.save(d/'cond.npy', cond_of(imgs)[0].float().cpu().numpy().astype(np.float16))
        json.dump({'uid': r['uid'], 'name': r['name'], 'license': r['license'],
                   'faces': int(len(m.faces)), 'chamfer': err}, open(d/'meta.json', 'w'))
        (d/'mesh.glb').unlink()                       # 원본 메시는 용량이 커서 지움
        stat['ok'] += 1
    except Exception:
        stat['error'] += 1
        if d.exists():
            for f in d.rglob('*'):
                if f.is_file(): f.unlink()
    if (i+1) % 5 == 0 or i == len(mine)-1:
        el = time.time()-t0
        log(f'{i+1}/{len(mine)}  {stat}  {el/(i+1):.1f}s/개  남음 {(len(mine)-i-1)*el/(i+1)/60:.0f}분')
log('완료', stat)
