#!/usr/bin/env bash
# 5단계(텍스처용 CUDA 확장)만 다시 빌드
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$REPO_DIR/hy3dgen/texgen/custom_rasterizer" && rm -rf build && python setup.py install
cd "$REPO_DIR/hy3dgen/texgen/differentiable_renderer" && rm -rf build && python setup.py install
cd "$HERE"
python - <<'PY'
import torch, hy3dgen, custom_rasterizer
print("torch", torch.__version__, "| cuda ok:", torch.cuda.is_available(), torch.cuda.device_count(), "GPU | custom_rasterizer OK")
PY
