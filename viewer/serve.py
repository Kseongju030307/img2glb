"""img2glb 결과 GLB 웹 뷰어 서버. Blender 없이 브라우저로 확인.

    source ~/img2glb/env.sh && python ~/img2glb/viewer/serve.py [--port 8766]
→ http://<서버IP>:8766/viewer/   (outputs/ 아래 GLB 목록이 드롭다운에 뜸)
"""
import argparse
import functools
import http.server
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8766)
ap.add_argument("--host", default="0.0.0.0")
a = ap.parse_args()
root = Path(__file__).resolve().parents[1]           # ~/img2glb
out_root = (root / "outputs").resolve()


class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/list"):
            items = []
            if out_root.is_dir():
                for g in sorted(out_root.glob("*/*.glb")):
                    items.append({"name": f"{g.parent.name}/{g.name}",
                                  "url": f"/outputs/{g.parent.name}/{g.name}",
                                  "mb": round(g.stat().st_size / 1e6, 1)})
            body = json.dumps(items).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/viewer", "/viewer/"):
            self.send_response(302); self.send_header("Location", "/viewer/index.html"); self.end_headers(); return
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "/api/" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


# outputs 가 /data 로 가는 symlink 라도 따라가도록 root 를 그대로 서빙 (follow_symlinks 기본 True)
handler = functools.partial(H, directory=str(root))
http.server.ThreadingHTTPServer.allow_reuse_address = True
print(f"serving {root} on http://{a.host}:{a.port}/viewer/")
http.server.ThreadingHTTPServer((a.host, a.port), handler).serve_forever()
