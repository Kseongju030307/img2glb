#!/usr/bin/env bash
# inputs/ 아래 캐릭터 폴더 여러 개를 GPU 4장에 나눠서 병렬 처리
# 사용: bash batch.sh            (inputs/* 전부)
#       bash batch.sh charA charB (지정 폴더만)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
source "$HERE/env.sh"
NGPU="${NGPU:-4}"

if [ $# -gt 0 ]; then
  TARGETS=("$@")
else
  mapfile -t TARGETS < <(find inputs -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

i=0
for name in "${TARGETS[@]}"; do
  gpu=$(( i % NGPU ))
  echo "[gpu $gpu] $name"
  CUDA_VISIBLE_DEVICES=$gpu python generate.py --input_dir "inputs/$name" --out "outputs/$name" \
    > "outputs/${name}.log" 2>&1 &
  i=$(( i + 1 ))
  # GPU 수만큼 채워지면 대기
  if (( i % NGPU == 0 )); then wait; fi
done
wait
echo "all done -> outputs/"
