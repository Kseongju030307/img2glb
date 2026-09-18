#!/usr/bin/env bash
# Hunyuan3D-2mv 설치 (RTX 3090 / CUDA 12.4). 모델·venv·repo 는 전부 /data/sungju030307 아래에 둠.
# 사용: bash setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

echo "[1/5] venv: $VENV  (python 3.12, /opt/anaconda3 기반 — 기존 *-venv 와 동일 방식)"
if [ ! -f "$VENV/bin/activate" ]; then
  /opt/anaconda3/bin/python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip ninja pybind11

echo "[2/5] PyTorch 2.6 (cu124)"
pip install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] Hunyuan3D-2 clone -> $REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$REPO_DIR"
fi
ln -sfn "$REPO_DIR" "$HERE/Hunyuan3D-2"
ln -sfn "$OUT_ROOT" "$HERE/outputs"

echo "[4/5] python deps"
cd "$REPO_DIR"
pip install -r requirements.txt
pip install -e .

bash "$HERE/patches/apply.sh"

echo "[5/5] 텍스처용 CUDA 확장 빌드"
cd "$REPO_DIR/hy3dgen/texgen/custom_rasterizer" && python setup.py install
cd "$REPO_DIR/hy3dgen/texgen/differentiable_renderer" && python setup.py install

cd "$HERE"
python - <<'PY'
import torch, hy3dgen
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "GPU")
import custom_rasterizer  # noqa
print("custom_rasterizer OK")
PY
echo
echo "설치 완료. 실행:"
echo "  source ~/img2glb/env.sh"
echo "  python ~/img2glb/generate.py --input_dir inputs/example --out outputs/example"
