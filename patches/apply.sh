#!/usr/bin/env bash
# 최신 diffusers 호환 패치. setup.sh 에서 자동 호출. 여러 번 실행해도 안전.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../env.sh"
F="$REPO_DIR/hy3dgen/texgen/utils/multiview_utils.py"
if ! grep -q "trust_remote_code=True" "$F"; then
  sed -i 's/custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16)/custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16, trust_remote_code=True)/' "$F"
  echo "patched: $F (trust_remote_code)"
else
  echo "already patched: $F"
fi
