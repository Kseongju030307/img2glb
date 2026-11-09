"""img2glb 웹 뷰어 + 작업 실행 서버.

    source ~/img2glb/env.sh && python ~/img2glb/viewer/serve.py [--port 8766]
→ http://<서버IP>:8766/viewer/

기능
  * GLB 보기: generated/03_glb, outputs 아래 GLB 목록 / 드래그 앤 드롭
  * GLB 만들기: 이미지 폴더 선택 → generate.py (형상 + 텍스처) [+ 고해상 색 전사] + 4방향 렌더
  * 텍스처 다시 굽기: GLB + 이미지 폴더 선택 → generate.py --mesh (형상 유지, 텍스처만)
작업은 systemd-run --user 유닛으로 띄워서 서버/세션이 꺼져도 계속 돈다. 로그·상태: /data/sungju030307/img2glb/logs/viewer_jobs/
"""
import argparse, functools, http.server, json, os, re, shlex, shutil, subprocess, time, urllib.parse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--host", default="0.0.0.0")
a = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]                      # ~/img2glb
GLB_DIR = ROOT / "generated" / "03_glb"
JOB_DIR = Path("/data/sungju030307/img2glb/logs/viewer_jobs"); JOB_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path("/data/sungju030307/img2glb/uploads"); UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# 탐색/파일 제공을 허용하는 루트 (그 밖의 경로는 거부)
ALLOWED = [Path.home().resolve(), Path("/data/sungju030307").resolve()]
VIEWS = ["front", "back", "left", "right"]
EXTRA = ["top", "bottom"]
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MIN_FREE_GB = 0.6                                                # 이보다 디스크 여유가 적으면 작업 거부
GPU_PREF = [1, 2, 3, 0]                                          # auto 일 때 비어 있는 GPU 를 이 순서로


def allowed(p: Path) -> bool:
    try:
        rp = p.resolve()
    except Exception:
        return False
    return any(rp == r or r in rp.parents for r in ALLOWED)


def detect_views(d: Path):
    """front/back/left/right(+top/bottom) 파일 찾기. front.png, character_front.png, front-1.jpg 등 허용."""
    found = {}
    files = [f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in IMG_EXT] if d.is_dir() else []
    for v in VIEWS + EXTRA:
        exact = [f for f in files if f.stem.lower() == v]
        loose = [f for f in files if re.search(rf"(^|[_\-\s.]){v}($|[_\-\s.])", f.stem.lower())]
        pick = (exact or loose or [None])[0]
        if pick: found[v] = str(pick)
    return found


def disk_free_gb(p: Path):
    q = p
    while not q.exists(): q = q.parent
    return shutil.disk_usage(q).free / 2**30


def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return [dict(index=int(i), mem_mb=int(m), util=int(u)) for i, m, u in (l.split(", ") for l in out.strip().splitlines())]
    except Exception:
        return []


def busy_gpus_by_jobs():
    return {j["gpu"] for j in list_jobs() if j["status"] == "running"}


def pick_gpu(req):
    if req not in (None, "", "auto"):
        return int(req)
    busy = busy_gpus_by_jobs()
    st = {g["index"]: g for g in gpu_state()}
    for g in GPU_PREF:
        if g in st and st[g]["mem_mb"] < 1000 and g not in busy:
            return g
    return None


# ───────────── 작업 ─────────────
def job_path(jid): return JOB_DIR / f"{jid}.json"


def stage_of(log: str):
    if re.search(r"종료 코드 0\s*$", log.strip()): return "완료"
    m = re.search(r"종료 코드 (\d+)", log)
    if m: return "취소됨" if m.group(1) == "130" else f"실패 (코드 {m.group(1)})"
    steps = [("[단계] 렌더", "4방향 렌더"), ("[단계] 고해상 색 전사", "색 전사"), ("texture 완료", "텍스처 완료"), ("== texture", "텍스처 굽는 중"),
             ("shape 완료", "형상 완료"), ("shape 건너뜀", "기존 메시 로드"), ("== shape", "형상 생성 중"),
             ("[단계] GLB 생성", "3D 생성 준비"), ("[단계] 4뷰", "4뷰 턴어라운드 생성 중"), ("== 입력", "배경 제거")]
    for key, name in steps:
        if key in log: return name
    return "모델 로딩"


def list_jobs():
    jobs = []
    for f in sorted(JOB_DIR.glob("*.json")):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        log = Path(j["log"]).read_text(errors="replace") if Path(j["log"]).exists() else ""
        st = stage_of(log)
        j["stage"] = st
        j["status"] = "done" if st == "완료" else "fail" if st.startswith("실패") or st == "취소됨" else "running"
        if j["status"] == "running":
            act = subprocess.run(["systemctl", "--user", "is-active", j["unit"]], capture_output=True, text=True).stdout.strip()
            if act not in ("active", "activating"):
                j["status"], j["stage"] = "fail", f"중단됨 ({act})"
        j["elapsed"] = int((j.get("ended") or time.time()) - j["started"])
        j["log_tail"] = "\n".join(l for l in log.replace("\r", "\n").splitlines() if l.strip() and "it/s" not in l and "s/it" not in l)[-1500:]
        jobs.append(j)
    return jobs[::-1]


def launch(kind, cmds, out_dir: Path, gpu, extra):
    jid = time.strftime("%m%d_%H%M%S") + f"_{kind}"
    log = JOB_DIR / f"{jid}.log"; unit = f"img2glb-{jid}".replace("_", "-")
    script = JOB_DIR / f"{jid}.sh"
    body = ["#!/bin/bash", f"cd {shlex.quote(str(ROOT))} && source env.sh && unset CUDA_VISIBLE_DEVICES",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "rc=0"]
    for title, c in cmds:
        body.append(f'echo "== [단계] {title}"')
        body.append(f"{c} || {{ rc=$?; echo \"[$(date +%H:%M:%S)] 종료 코드 $rc\"; exit $rc; }}")
    body.append('echo "[$(date +%H:%M:%S)] 종료 코드 0"')
    script.write_text("\n".join(body) + "\n"); script.chmod(0o755)
    subprocess.run(["systemd-run", "--user", f"--unit={unit}", "--collect", "bash", "-lc", f"{shlex.quote(str(script))} > {shlex.quote(str(log))} 2>&1"],
                   check=True, capture_output=True)
    rel = out_dir.relative_to(ROOT) if ROOT in out_dir.parents else None
    j = dict(id=jid, kind=kind, unit=unit, gpu=gpu, log=str(log), out=str(out_dir), started=time.time(), **extra,
             result_url=f"/api/file?path={urllib.parse.quote(str(out_dir / 'mesh_textured.glb'))}",
             views_url=f"/api/file?path={urllib.parse.quote(str(out_dir / 'mesh_textured_views.png'))}")
    job_path(jid).write_text(json.dumps(j, ensure_ascii=False))
    return j


def common_paint_args(p):
    s = f" --paint_subfolder hunyuan3d-paint-v2-0 --tex_faces {int(p.get('tex_faces', 300000))}"
    if p.get("oblique", True): s += " --paint_oblique"
    return s


def view_args(views):
    return "".join(f" --{v} {shlex.quote(views[v])}" for v in VIEWS if v in views)


def start_generate(p):
    d = Path(p["input_dir"]).expanduser()
    if not allowed(d) or not d.is_dir(): raise ValueError(f"폴더를 열 수 없음: {d}")
    views = detect_views(d)
    if "front" not in views: raise ValueError("front 이미지가 없음 (front.png 또는 *_front.png)")
    name = re.sub(r"[^\w\-.]", "_", (p.get("name") or d.name).strip()) or d.name
    out = GLB_DIR / name
    if out.exists() and not p.get("overwrite"): raise ValueError(f"이미 있는 결과 폴더: {out.name} — '덮어쓰기'를 체크하거나 이름을 바꾸세요")
    if disk_free_gb(GLB_DIR) < MIN_FREE_GB: raise ValueError(f"디스크 여유 {disk_free_gb(GLB_DIR):.2f}GB — 너무 적음")
    gpu = pick_gpu(p.get("gpu"))
    if gpu is None: raise ValueError("비어 있는 GPU 가 없음")
    q = shlex.quote; o = q(str(out))
    cmds = []
    if out.exists() and p.get("overwrite"): cmds.append(("기존 결과 삭제", f"rm -rf {o}"))
    cmds.append(("GLB 생성", f"python generate.py --input_dir {q(str(d))}{view_args(views)} --out {o} --octree {int(p.get('octree', 1024))} --flashvdm"
                 f"{common_paint_args(p)} --gpu {gpu}"))
    if p.get("transfer"):
        cmds.append(("고해상 색 전사", f"python transfer_texture.py --hi {o}/mesh_white.glb --lo {o}/mesh_textured.glb --out {o}/mesh_white_textured.glb"))
    cmds.append(("렌더", f"python bench/render_tex.py {o}/mesh_textured.glb {o}/mesh_textured_views.png"))
    return launch("generate", cmds, out, gpu, dict(title=f"GLB 만들기: {name}", input_dir=str(d), views=views))


def start_retexture(p):
    g = Path(p["glb"]).expanduser(); d = Path(p["input_dir"]).expanduser()
    if not allowed(g) or not g.is_file(): raise ValueError(f"GLB 를 열 수 없음: {g}")
    if not allowed(d) or not d.is_dir(): raise ValueError(f"이미지 폴더를 열 수 없음: {d}")
    views = detect_views(d)
    if "front" not in views: raise ValueError("front 이미지가 없음 (front.png 또는 *_front.png)")
    # 출력: 비우면 GLB 가 있는 폴더(그 안의 mesh_textured.glb 를 덮어씀), 이름을 주면 generated/03_glb/<이름>
    name = (p.get("name") or "").strip()
    out = GLB_DIR / re.sub(r"[^\w\-.]", "_", name) if name else g.parent
    if not allowed(out): raise ValueError(f"출력 폴더 불가: {out}")
    if disk_free_gb(out) < MIN_FREE_GB: raise ValueError(f"디스크 여유 {disk_free_gb(out):.2f}GB — 너무 적음")
    gpu = pick_gpu(p.get("gpu"))
    if gpu is None: raise ValueError("비어 있는 GPU 가 없음")
    q = shlex.quote; o = q(str(out))
    norm = ""
    if p.get("normalize"):                                         # STL/OBJ 등 외부 메시
        norm = f" --mesh_normalize --mesh_up {q(p.get('up', 'y'))}"
        if p.get("yaw"): norm += f" --mesh_yaw {float(p['yaw'])}"
    cmds = [("텍스처 굽기", f"python generate.py --input_dir {q(str(d))}{view_args(views)} --mesh {q(str(g))}{norm} --out {o}"
                         f"{common_paint_args(p)} --gpu {gpu}")]
    if p.get("transfer"):
        cmds.append(("고해상 색 전사", f"python transfer_texture.py --hi {q(str(g))} --lo {o}/mesh_textured.glb --out {o}/mesh_white_textured.glb"))
    cmds.append(("렌더", f"python bench/render_tex.py {o}/mesh_textured.glb {o}/mesh_textured_views.png"))
    return launch("retexture", cmds, out, gpu, dict(title=f"텍스처 다시 굽기: {g.parent.name}/{g.name}", input_dir=str(d), glb=str(g), views=views))


def start_auto(p):
    """이미지 1장 -> turnaround.py(4뷰 자동 생성) -> generate.py(3D) -> 렌더. 한 job 으로 체이닝."""
    front = Path(p["front"]).expanduser()
    if not allowed(front) or not front.is_file(): raise ValueError(f"이미지를 열 수 없음: {front}")
    name = re.sub(r"[^\w\-.]", "_", (p.get("name") or front.stem).strip()) or front.stem
    out = GLB_DIR / name
    if out.exists() and not p.get("overwrite"): raise ValueError(f"이미 있는 결과 폴더: {out.name} — '덮어쓰기'를 체크하거나 이름을 바꾸세요")
    if disk_free_gb(GLB_DIR) < MIN_FREE_GB: raise ValueError(f"디스크 여유 {disk_free_gb(GLB_DIR):.2f}GB — 너무 적음")
    gpu = pick_gpu(p.get("gpu"))
    if gpu is None: raise ValueError("비어 있는 GPU 가 없음")
    turn_gpus = str(p.get("turn_gpus") or "2,3").strip()           # Qwen 편집 모델은 2장 필요
    views = "front,back,left" if p.get("four_view") else "front,back"
    q = shlex.quote; o = q(str(out)); turn = out / "src_views"; t = q(str(turn))
    cmds = []
    if out.exists() and p.get("overwrite"): cmds.append(("기존 결과 삭제", f"rm -rf {o}"))
    ta = (f"python turnaround.py --front {q(str(front))} --out {t} --gpus {q(turn_gpus)} "
          f"--views {q(views)} --size {int(p.get('size', 1024))} --seed {int(p.get('seed', 1234))}")
    if not p.get("chibi", True): ta += " --no_chibi"
    if p.get("colors"): ta += f" --colors {q(p['colors'])}"
    elif p.get("quantize"): ta += f" --palette {int(p.get('n_colors', 5))}"
    cmds.append(("4뷰 턴어라운드 생성", ta))
    cmds.append(("GLB 생성", f"python generate.py --input_dir {t} --out {o} --octree {int(p.get('octree', 1024))} --flashvdm"
                 f"{common_paint_args(p)} --gpu {gpu}"))
    if p.get("transfer"):
        cmds.append(("고해상 색 전사", f"python transfer_texture.py --hi {o}/mesh_white.glb --lo {o}/mesh_textured.glb --out {o}/mesh_white_textured.glb"))
    cmds.append(("렌더", f"python bench/render_tex.py {o}/mesh_textured.glb {o}/mesh_textured_views.png"))
    extra = dict(title=f"이미지→3D: {name}", front=str(front), views=views, auto=True,
                 turn_url=f"/api/file?path={urllib.parse.quote(str(turn / 'sheet.png'))}")
    return launch("auto", cmds, out, gpu, extra)


def browse(path):
    p = Path(path or str(ROOT)).expanduser()
    if not allowed(p): p = ROOT
    if p.is_file(): p = p.parent
    dirs, files = [], []
    for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
        if c.name.startswith("."): continue
        try:
            if c.is_dir(): dirs.append(c.name)
            elif c.suffix.lower() in IMG_EXT | {".glb", ".stl", ".obj", ".ply"}:
                files.append(dict(name=c.name, kind="mesh" if c.suffix.lower() in {".glb", ".stl", ".obj", ".ply"} else "image",
                                  mb=round(c.stat().st_size / 1e6, 1), path=str(c)))
        except OSError:
            pass
    parent = str(p.parent) if allowed(p.parent) and p != p.parent else None
    return dict(path=str(p), parent=parent, dirs=dirs, files=files, views=detect_views(p),
                roots=[str(ROOT / "inputs"), str(GLB_DIR), str(Path.home()), "/data/sungju030307"])


def glb_list():
    items = []
    for base in (GLB_DIR, ROOT / "outputs"):
        if base.is_dir():
            for g in sorted(base.glob("*/*.glb")):
                items.append(dict(name=f"{g.parent.name}/{g.name}", path=str(g), mb=round(g.stat().st_size / 1e6, 1),
                                  url=f"/api/file?path={urllib.parse.quote(str(g))}"))
    return items


class H(http.server.SimpleHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); qs = urllib.parse.parse_qs(u.query)
        try:
            if u.path == "/api/list": return self._json(glb_list())
            if u.path == "/api/browse": return self._json(browse(qs.get("path", [""])[0]))
            if u.path == "/api/jobs": return self._json(list_jobs())
            if u.path == "/api/status":
                return self._json(dict(gpus=gpu_state(), free_gb=round(disk_free_gb(GLB_DIR), 2), busy=sorted(busy_gpus_by_jobs())))
            if u.path == "/api/file":
                f = Path(qs.get("path", [""])[0])
                if not allowed(f) or not f.is_file(): return self._json(dict(error="not found"), 404)
                ctype = {".glb": "model/gltf-binary", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                         ".webp": "image/webp"}.get(f.suffix.lower(), "application/octet-stream")
                self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(f.stat().st_size))
                self.end_headers()
                with open(f, "rb") as fh: shutil.copyfileobj(fh, self.wfile)
                return
        except Exception as e:
            return self._json(dict(error=str(e)), 400)
        if u.path in ("/", "/viewer", "/viewer/"):
            self.send_response(302); self.send_header("Location", "/viewer/index.html"); self.end_headers(); return
        return super().do_GET()

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0))
            if u.path == "/api/upload":                                # 이미지 원본 바이트 업로드
                data = self.rfile.read(n)
                if not data: raise ValueError("빈 업로드")
                if len(data) > 40 * 2**20: raise ValueError("파일이 너무 큼 (40MB 초과)")
                qs = urllib.parse.parse_qs(u.query)
                nm = re.sub(r"[^\w\-.]", "_", (qs.get("name", ["upload.png"])[0])) or "upload.png"
                if not os.path.splitext(nm)[1]: nm += ".png"
                dst = UPLOAD_DIR / (time.strftime("%m%d_%H%M%S_") + nm)
                dst.write_bytes(data)
                return self._json(dict(path=str(dst)))
            p = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/generate": return self._json(start_generate(p))
            if u.path == "/api/retexture": return self._json(start_retexture(p))
            if u.path == "/api/auto": return self._json(start_auto(p))
            if u.path == "/api/cancel":
                j = json.loads(job_path(p["id"]).read_text())
                subprocess.run(["systemctl", "--user", "stop", j["unit"]], capture_output=True)
                with open(j["log"], "a") as fh: fh.write(f"\n[{time.strftime('%H:%M:%S')}] 종료 코드 130 (취소)\n")
                return self._json(dict(ok=True))
            return self._json(dict(error="unknown"), 404)
        except Exception as e:
            return self._json(dict(error=str(e)), 400)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store"); super().end_headers()

    def log_message(self, fmt, *args):
        if "/api/" not in (args[0] if args else ""): super().log_message(fmt, *args)


handler = functools.partial(H, directory=str(ROOT))
http.server.ThreadingHTTPServer.allow_reuse_address = True
print(f"serving {ROOT} on http://{a.host}:{a.port}/viewer/")
http.server.ThreadingHTTPServer((a.host, a.port), handler).serve_forever()
