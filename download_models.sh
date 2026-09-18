#!/usr/bin/env bash
# 가중치 사전 다운로드 → /data/sungju030307/hf-cache  (첫 실행 시 자동으로도 받아짐)
# shape: model.fp16.safetensors (4.9GB) / texture: paint turbo + delight (~10GB)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("tencent/Hunyuan3D-2mv",
                      allow_patterns=["hunyuan3d-dit-v2-mv/config.yaml", "hunyuan3d-dit-v2-mv/model.fp16.safetensors"])
print("shape  ->", p)
p = snapshot_download("tencent/Hunyuan3D-2",
                      allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0-turbo/*"])
print("texture->", p)
PY
du -sh "$HF_HUB_CACHE"/models--tencent--*
