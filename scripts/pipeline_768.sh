#!/bin/bash
# 한 스크립트로: 768 턴어라운드(앞/뒤, 디자인 유지) -> 각 뷰 평면색 편집(flatten) -> 공유 5색.
# 1024 는 3장 분산에서 로딩/첫 스텝에서 멈춤(2회) → 768 이 현실적 상한.
G=/data/sungju030307/img2glb/logs; D=/data/sungju030307/img2glb/generated/02_turnaround_768
free3() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{print $1}' | tr '\n' ',' | sed 's/,$//'; }
stable=0; while true; do F=$(free3); N=$(echo "$F" | awk -F, '{print NF}'); [ "$N" -ge 3 ] && stable=$((stable+1)) || stable=0; [ "$stable" -ge 2 ] && break; sleep 30; done
GPUS=$(echo "$F" | tr ',' '\n' | head -3 | tr '\n' ',' | sed 's/,$//')
cd /home/sungju030307/img2glb && source env.sh && unset CUDA_VISIBLE_DEVICES; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date +%H:%M:%S)] 1/2 턴어라운드 768 (GPU $GPUS)"
python turnaround.py --front inputs/smaple4/front.png --gpus $GPUS --steps 12 --cfg 3.0 --size 768 --out $D > $G/turn_768.log 2>&1 || { echo "턴어라운드 실패 $?"; exit 1; }
echo "[$(date +%H:%M:%S)] 2/2 평면색 편집 + 5색"
python flatten.py --gpus $GPUS --imgs $D/front.png $D/back.png --n 1 --size 768 --out $D/flat > $G/flatten_768.log 2>&1
echo "[$(date +%H:%M:%S)] 종료 코드 $?"
