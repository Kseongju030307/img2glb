# 공통 경로 설정. 모든 스크립트가 이 파일을 source 함.
#   source ~/img2glb/env.sh   → venv 활성화 + 캐시 경로 설정
DATA_ROOT="${DATA_ROOT:-/data/sungju030307}"          # 2TB 디스크
IMG2GLB_DATA="$DATA_ROOT/img2glb"                     # repo / outputs
VENV="$DATA_ROOT/img2glb-venv"                        # python venv (기존 *-venv 관례)
REPO_DIR="$IMG2GLB_DATA/Hunyuan3D-2"
OUT_ROOT="$IMG2GLB_DATA/outputs"

export HF_HUB_CACHE="$DATA_ROOT/hf-cache"             # 기존 HF 캐시 (models--* 가 바로 이 아래)
export HF_HOME="$DATA_ROOT/hf-home"
export PIP_CACHE_DIR="$DATA_ROOT/pip-cache"
export TORCH_HOME="$DATA_ROOT/torch-cache"
export U2NET_HOME="$DATA_ROOT/u2net"                  # rembg 가중치
export TMPDIR="$IMG2GLB_DATA/tmp"
export PYTHONUNBUFFERED=1                             # 로그 실시간 출력
# CUDA 확장 빌드용: anaconda 의 nvcc(헤더 없음) 대신 시스템 CUDA 사용
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"                     # RTX 3090
mkdir -p "$IMG2GLB_DATA" "$OUT_ROOT" "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$U2NET_HOME" "$TMPDIR"

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
