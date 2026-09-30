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

VIEWS = ["front", "back", "left", "right"]          # 형상(Hunyuan3D-2mv) 은 이 4뷰만 받는다 (view_num=4 고정)
PAINT_ONLY_VIEWS = ["top", "bottom"]                  # 텍스처(Paint) 참조로만 쓰는 추가 뷰. 있으면 자동 포함
EXTS = [".png", ".jpg", ".jpeg", ".webp"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("입력")
    src.add_argument("--input_dir", type=str, help="front/back/left/right.png 가 들어있는 폴더")
    for v in VIEWS:
        src.add_argument(f"--{v}", type=str, help=f"{v} 뷰 이미지 경로 (input_dir 보다 우선)")
    p.add_argument("--out", type=str, required=True, help="출력 폴더")
    p.add_argument("--mesh_normalize", action="store_true",
                   help="--mesh 로 넣은 외부 메시(Meshy 등)를 원점 중심·크기 1.0 으로 맞춤. Hunyuan 출력이 아니면 켤 것")
    p.add_argument("--mesh_up", choices=["y", "z"], default="y",
                   help="--mesh 의 위쪽 축. STL/CAD 는 보통 z. (mesh_normalize 와 함께 적용)")
    p.add_argument("--mesh_yaw", type=float, default=0.0,
                   help="--mesh_normalize 시 Y축 회전(도). 정면이 안 맞을 때 90/180/-90 로 조정")
    p.add_argument("--mesh", type=str, default=None,
                   help="이미 만든 메시 경로(GLB/STL/OBJ/PLY, trimesh 가 읽는 것 전부). 주면 shape 단계를 건너뛰고 텍스처만 입힘")

    g = p.add_argument_group("shape (Hunyuan3D-2mv)")
    g.add_argument("--shape_model", default="tencent/Hunyuan3D-2mv")
    g.add_argument("--shape_subfolder", default="hunyuan3d-dit-v2-mv")
    g.add_argument("--steps", type=int, default=50, help="diffusion step 수 (기본 50)")
    g.add_argument("--octree", type=int, default=380, help="octree 해상도. 클수록 디테일↑ 시간↑ (256~512)")
    g.add_argument("--num_chunks", type=int, default=20000)
    g.add_argument("--flashvdm", action="store_true",
                   help="FlashVDM 디코더 사용. 표면 근처만 촘촘히 평가해서 고해상도(octree 1024)를 "
                        "dense 격자보다 훨씬 싸게 뽑는다. turbo VAE 로 교체됨")
    g.add_argument("--seed", type=int, default=12345)
    g.add_argument("--guidance", type=float, default=5.0)

    t = p.add_argument_group("texture (Hunyuan3D-Paint)")
    t.add_argument("--no_texture", action="store_true", help="텍스처 생성 생략 (메시만)")
    t.add_argument("--paint_model", default="tencent/Hunyuan3D-2")
    t.add_argument("--paint_subfolder", default="hunyuan3d-paint-v2-0-turbo",
                   help="hunyuan3d-paint-v2-0 (느림/고품질) 또는 hunyuan3d-paint-v2-0-turbo")
    t.add_argument("--paint_bake_exp", type=float, default=None,
                   help="베이킹 cos 지수. 기본 4 는 뷰 경계가 칼같이 갈라져 얼룩이 됨. 2 로 낮추면 경계가 부드럽게 섞임")
    t.add_argument("--paint_side_weight", type=float, default=None,
                   help="좌/우 뷰 베이킹 가중치. 파이프라인 기본 0.1 은 옆면에 앞/뒤 투영이 섞여 얼룩이 생김. 0.5 권장")
    t.add_argument("--paint_views", default="front,back,left,right,top,bottom",
                   help="텍스처 참조로 넣을 뷰 (쉼표). 파일이 없는 뷰는 건너뜀. top/bottom 은 텍스처에만 쓰임")
    t.add_argument("--unlit", action="store_true",
                   help="KHR_materials_unlit 로 내보냄. 뷰어 조명을 안 받아 텍스처 색 그대로(셀 셰이딩/평면). "
                        "안 주면 metallic=0 만 적용 (조명은 받되 금속처럼 어둡게 눌리지 않음)")
    t.add_argument("--paint_oblique", action="store_true",
                   help="베이킹 카메라에 el±20 비스듬한 뷰 8장 추가. 정수리/바닥 정면(el±90) 생성이 저품질이라 "
                        "그 경계 영역을 품질 좋은 비스듬한 뷰가 맡게 한다. 뷰 6->14 라 페인트 시간 ~2배")
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
        for v in VIEWS + PAINT_ONLY_VIEWS:
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


def fix_glb_material(path, unlit=False):
    """trimesh 가 metallicFactor 를 안 써서 뷰어가 기본값 1.0(완전 금속)으로 본다.
    그러면 텍스처가 어둡게 눌리고 반사가 생겨 음영처럼 보인다. 캐릭터는 0 이 맞다.
    unlit=True 면 KHR_materials_unlit 도 붙여 뷰어 조명을 아예 안 받게 한다."""
    import json, struct
    with open(path, "rb") as f:
        magic, ver, _ = struct.unpack("<III", f.read(12))
        jl, jt = struct.unpack("<II", f.read(8)); j = json.loads(f.read(jl)); rest = f.read()
    for m in j.setdefault("materials", [{}]):
        pbr = m.setdefault("pbrMetallicRoughness", {}); pbr["metallicFactor"] = 0.0; pbr["roughnessFactor"] = 1.0
        if unlit:
            m.setdefault("extensions", {})["KHR_materials_unlit"] = {}
            j["extensionsUsed"] = sorted(set(j.get("extensionsUsed", [])) | {"KHR_materials_unlit"})
    jb = json.dumps(j, separators=(",", ":")).encode(); jb += b" " * ((4 - len(jb) % 4) % 4)
    body = struct.pack("<II", len(jb), jt) + jb + rest
    with open(path, "wb") as f:
        f.write(struct.pack("<III", magic, ver, 12 + len(body)) + body)
    print(f"   재질 보정: metallic=0" + (", unlit" if unlit else ""))


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
        # 외부(Meshy 등) 메시는 좌표계/크기가 제각각이다. Paint 는 고정 카메라로 메시를 렌더해
        # 입력 뷰와 맞추므로, Hunyuan 출력과 같은 프레임(원점 중심, 최대 반경 ~1, Y-up, 정면 +Z)으로 맞춘다.
        if args.mesh_normalize:
            import numpy as np
            # STL 은 삼각형마다 정점이 따로 있어 법선이 각져 보인다(Paint 가 법선맵을 렌더하므로 영향).
            # 정점을 합쳐 매끈한 법선을 얻는다. GLB 에도 무해.
            mesh.merge_vertices()
            if args.mesh_up == "z":   # CAD/STL 은 Z-up 이 흔함 -> Y-up 으로
                mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
            mesh.apply_translation(-mesh.bounds.mean(axis=0))
            mesh.apply_scale(1.0 / np.abs(mesh.vertices).max())
            if args.mesh_yaw:
                mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(args.mesh_yaw), [0, 1, 0]))
            print(f"   정규화: 중심 이동 + 크기 1.0, yaw {args.mesh_yaw}°  bbox={np.round(mesh.extents, 3)}")
        if args.reduce_faces > 0:
            mesh = FaceReducer()(mesh, max_facenum=args.reduce_faces)
            print(f"   면 수 축소 -> {len(mesh.faces)}")
    else:
        print(f"== shape: {args.shape_model}/{args.shape_subfolder}")
        t0 = time.time()
        shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.shape_model, subfolder=args.shape_subfolder, variant="fp16"
        )
        if args.flashvdm:
            # octree 1024 는 dense 격자가 (1024+1)^3 = 10억 점이라 512 의 8배다.
            # FlashVDM 은 표면 근처만 계층적으로 평가해서 이걸 감당 가능하게 만든다.
            shape_pipe.enable_flashvdm(enabled=True)
            print("   FlashVDM 디코더 활성화 (turbo VAE)")
        mesh = shape_pipe(
            image={v: images[v] for v in VIEWS if v in images},   # top/bottom 제외
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
        if args.paint_bake_exp is not None:
            paint_pipe.config.bake_exp = args.paint_bake_exp
            print(f"   페인트 bake_exp -> {args.paint_bake_exp}")
        if args.paint_side_weight is not None:
            # candidate_view_weights = [앞, 좌, 뒤, 우, 위, 아래] 순. 기본 [1, .1, .5, .1, .05, .05]
            w = list(paint_pipe.config.candidate_view_weights)
            w[1] = w[3] = args.paint_side_weight
            paint_pipe.config.candidate_view_weights = w
            print(f"   페인트 뷰 가중치 -> {w}")
        if args.paint_oblique:
            cfg = paint_pipe.config
            az = list(cfg.candidate_camera_azims); el = list(cfg.candidate_camera_elevs); w = list(cfg.candidate_view_weights)
            for e in (20, -20):
                for a_ in (0, 90, 180, 270):
                    az.append(a_); el.append(e); w.append(0.3)
            cfg.candidate_camera_azims, cfg.candidate_camera_elevs, cfg.candidate_view_weights = az, el, w
            print(f"   페인트 비스듬한 뷰 추가 -> 총 {len(az)}뷰")
        wanted = [v.strip() for v in args.paint_views.split(",") if v.strip()]
        paint_imgs = [images[v] for v in wanted if v in images]
        print(f"   paint 입력 뷰: {[v for v in wanted if v in images]}")
        mesh = paint_pipe(mesh, image=paint_imgs if len(paint_imgs) > 1 else paint_imgs[0])
        tex_path = out / "mesh_textured.glb"
        mesh.export(tex_path)
        fix_glb_material(tex_path, unlit=args.unlit)
        result["textured"] = str(tex_path)
        print(f"   texture 완료 {time.time() - t0:.1f}s  저장: {tex_path}")

    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("== done:", out)


if __name__ == "__main__":
    main()
