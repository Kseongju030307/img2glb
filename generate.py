#!/usr/bin/env python
"""
2D 이미지 (front / back / left / right, 최대 4장) -> 텍스처 포함 GLB

사용 예:
  python generate.py --input_dir inputs/example --out outputs/example
  python generate.py --front f.png --back b.png --left l.png --right r.png --out outputs/char1
  python generate.py --input_dir inputs/example --out outputs/example --no_texture   # 메시만 (흰색)

input_dir 안에는 front.png / back.png / left.png / right.png 이름으로 넣으면 됨 (front 필수, 나머지 선택).
배경이 있는 이미지는 자동으로 rembg 로 배경 제거함 (RGBA 투명 배경이면 그대로 사용).

출력:
  <out>/mesh_white.glb      : 형태만 (텍스처 없음)
  <out>/mesh_textured.glb   : 텍스처 포함 (--no_texture 아니면)
  <out>/inputs/*.png        : 실제 파이프라인에 들어간 전처리 이미지 (확인용)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # bgremove.py

import faulthandler, signal
faulthandler.register(signal.SIGUSR1, all_threads=False)  # 멈춘 것 같으면: kill -USR1 <pid> → stderr 에 스택 출력

VIEWS = ["front", "back", "left", "right"]
EXTS = [".png", ".jpg", ".jpeg", ".webp"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("입력")
    src.add_argument("--input_dir", type=str, help="front/back/left/right.png 가 들어있는 폴더")
    for v in VIEWS:
        src.add_argument(f"--{v}", type=str, help=f"{v} 뷰 이미지 경로 (input_dir 보다 우선)")
    p.add_argument("--out", type=str, required=True, help="출력 폴더")
    p.add_argument("--mesh", type=str, default=None,
                   help="이미 만든 GLB 경로. 주면 shape 단계를 건너뛰고 텍스처만 입힘 (예: outputs/x/mesh_white.glb)")

    g = p.add_argument_group("shape (Hunyuan3D-2mv)")
    g.add_argument("--shape_model", default="tencent/Hunyuan3D-2mv")
    g.add_argument("--shape_subfolder", default="hunyuan3d-dit-v2-mv")
    g.add_argument("--steps", type=int, default=50, help="diffusion step 수 (기본 50)")
    g.add_argument("--octree", type=int, default=380, help="octree 해상도. 클수록 디테일↑ 시간↑ (256~512)")
    g.add_argument("--num_chunks", type=int, default=20000)
    g.add_argument("--seed", type=int, default=12345)
    g.add_argument("--guidance", type=float, default=5.0)

    t = p.add_argument_group("texture (Hunyuan3D-Paint)")
    t.add_argument("--no_texture", action="store_true", help="텍스처 생성 생략 (메시만)")
    t.add_argument("--paint_model", default="tencent/Hunyuan3D-2")
    t.add_argument("--paint_subfolder", default="hunyuan3d-paint-v2-0-turbo",
                   help="hunyuan3d-paint-v2-0 (느림/고품질) 또는 hunyuan3d-paint-v2-0-turbo")
    t.add_argument("--paint_views", default="front,back,left,right",
                   help="텍스처 생성에 넣을 뷰 (쉼표). 'front' 만 넣으면 공식 예제와 동일")
    t.add_argument("--tex_faces", type=int, default=100000,
                   help="텍스처 전에 면 수를 이 값으로 축소 (UV 전개가 CPU 라 100만 면이면 수십 분 걸림). 0 이면 축소 안 함. 공식 앱은 40000")

    m = p.add_argument_group("후처리")
    m.add_argument("--no_postprocess", action="store_true", help="floater 제거 / 퇴화 면 제거 생략")
    m.add_argument("--reduce_faces", type=int, default=0, help="0 이면 안 함. 예: 40000 -> 면 수 축소 (texture 전에 적용)")

    p.add_argument("--bg", default="auto", choices=["auto", "key", "rembg"],
                   help="배경 제거 방식. auto: 단색 배경이면 색 키잉, 아니면 rembg (RGBA 입력은 그대로)")
    p.add_argument("--bg_tol", type=int, default=14, help="배경 키잉 허용 오차 (기본 14). 바닥 그림자가 남으면 30~50")
    p.add_argument("--gpu", type=int, default=None, help="사용할 GPU 번호 (3090 4장 중). 안 주면 CUDA_VISIBLE_DEVICES 그대로")
    p.add_argument("--low_vram", action="store_true", help="모델을 필요할 때만 GPU 에 올림")
    return p.parse_args()


def find_view(input_dir: Path, view: str):
    for ext in EXTS:
        f = input_dir / f"{view}{ext}"
        if f.exists():
            return f
    return None


def collect_inputs(args):
    paths = {}
    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            sys.exit(f"[err] input_dir 없음: {d}")
        for v in VIEWS:
            f = find_view(d, v)
            if f:
                paths[v] = f
    for v in VIEWS:
        explicit = getattr(args, v)
        if explicit:
            paths[v] = Path(explicit)
    if "front" not in paths:
        sys.exit("[err] front 이미지는 필수입니다 (front.png 또는 --front)")
    return paths


def load_and_clean(paths, save_dir: Path, bg_mode="auto", bg_tol=14):
    """배경 제거 (bgremove.py). 전처리 결과를 save_dir 에 저장."""
    from PIL import Image
    from bgremove import remove_background

    save_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    for v, f in paths.items():
        img = remove_background(Image.open(f), bg_mode, bg_tol)
        images[v] = img
        img.save(save_dir / f"{v}.png")
        print(f"  [{v}] {f}  -> {img.size} {img.mode}")
    return images


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # 캐시 경로: env.sh 를 source 했으면 이미 설정됨. 안 했을 때의 기본값.
    os.environ.setdefault("HF_HUB_CACHE", "/data/sungju030307/hf-cache")
    os.environ.setdefault("U2NET_HOME", "/data/sungju030307/u2net")

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = collect_inputs(args)

    print(f"== 입력 ({len(paths)}장)")
    images = load_and_clean(paths, out / "inputs", args.bg, args.bg_tol)

    # ---------- 1) shape ----------
    if args.mesh:
        import trimesh
        white_path = Path(args.mesh)
        mesh = trimesh.load(white_path, force="mesh")
        print(f"== shape 건너뜀, 기존 메시 사용: {white_path}  faces={len(mesh.faces)}")
        if args.reduce_faces > 0:
            mesh = FaceReducer()(mesh, max_facenum=args.reduce_faces)
            print(f"   면 수 축소 -> {len(mesh.faces)}")
    else:
        print(f"== shape: {args.shape_model}/{args.shape_subfolder}")
        t0 = time.time()
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.shape_model, subfolder=args.shape_subfolder, variant="fp16"
        )
        mesh = shape_pipe(
            image=images,
            num_inference_steps=args.steps,
            octree_resolution=args.octree,
            num_chunks=args.num_chunks,
            guidance_scale=args.guidance,
            generator=torch.manual_seed(args.seed),
            output_type="trimesh",
        )[0]
        print(f"   shape 완료 {time.time() - t0:.1f}s  faces={len(mesh.faces)}")

        if not args.no_postprocess:
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
        if args.reduce_faces > 0:
            mesh = FaceReducer()(mesh, max_facenum=args.reduce_faces)
            print(f"   면 수 축소 -> {len(mesh.faces)}")

        white_path = out / "mesh_white.glb"
        mesh.export(white_path)
        print(f"   저장: {white_path}")

        if args.low_vram:
            del shape_pipe
            torch.cuda.empty_cache()

    # ---------- 2) texture ----------
    result = {"inputs": {k: str(v) for k, v in paths.items()}, "white": str(white_path), "textured": None,
              "args": vars(args)}
    if not args.no_texture:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        print(f"== texture: {args.paint_model}/{args.paint_subfolder}")
        t0 = time.time()
        if args.tex_faces > 0 and len(mesh.faces) > args.tex_faces:
            mesh = FaceReducer()(mesh, max_facenum=args.tex_faces)
            print(f"   텍스처용 면 수 축소 -> {len(mesh.faces)}  (mesh_white.glb 는 원본 유지)")
        paint_pipe = Hunyuan3DPaintPipeline.from_pretrained(args.paint_model, subfolder=args.paint_subfolder)
        wanted = [v.strip() for v in args.paint_views.split(",") if v.strip()]
        paint_imgs = [images[v] for v in wanted if v in images]
        print(f"   paint 입력 뷰: {[v for v in wanted if v in images]}")
        mesh = paint_pipe(mesh, image=paint_imgs if len(paint_imgs) > 1 else paint_imgs[0])
        tex_path = out / "mesh_textured.glb"
        mesh.export(tex_path)
        result["textured"] = str(tex_path)
        print(f"   texture 완료 {time.time() - t0:.1f}s  저장: {tex_path}")

    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("== done:", out)


if __name__ == "__main__":
    main()
