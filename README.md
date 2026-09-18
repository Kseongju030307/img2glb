# img2glb — 2D 이미지 4장 → 텍스처 포함 3D GLB (Hunyuan3D-2mv)

front / back / left / right 이미지를 넣으면
**Hunyuan3D-2mv** 로 메시를 만들고 **Hunyuan3D-Paint** 로 텍스처를 입혀 `.glb` 로 내보냅니다.

```
~/img2glb/                        (코드만, 홈 디스크는 거의 안 씀)
├── env.sh               # 공통 경로 설정 + venv 활성화  ← 항상 먼저 source
├── setup.sh             # 1회 설치 (venv + Hunyuan3D-2 + CUDA 확장 빌드)
├── download_models.sh   # 가중치 미리 받기 (선택)
├── generate.py          # 메인 스크립트: 이미지 → mesh_white.glb + mesh_textured.glb
├── bgremove.py          # 배경 제거 (단색 배경이면 색 키잉, 아니면 rembg). generate.py 가 자동 사용
├── shade.py             # (선택) 플랫 컬러 그림에 가짜 음영 추가. 보통 필요 없음
├── batch.sh             # inputs/ 폴더들을 GPU 4장에 나눠 병렬 처리
├── build_ext.sh         # CUDA 확장만 다시 빌드 (5단계 재실행)
├── patches/apply.sh     # 최신 diffusers 호환 패치 (setup.sh 가 자동 적용)
├── inputs/<name>/       # front.png back.png left.png right.png
├── outputs -> /data/sungju030307/img2glb/outputs      (symlink)
└── Hunyuan3D-2 -> /data/sungju030307/img2glb/Hunyuan3D-2 (symlink)

/data/sungju030307/               (2TB 디스크, 용량 큰 것 전부 여기)
├── img2glb-venv/        # python venv (기존 *-venv 와 같은 방식)
├── img2glb/Hunyuan3D-2  # 공식 repo
├── img2glb/outputs/     # 결과
├── hf-cache/            # HF 가중치 (기존 캐시 공유, models--tencent--Hunyuan3D-2*)
├── pip-cache/           # pip 캐시 (기존)
└── u2net/               # rembg 가중치
```

## 1. 설치 (1회)

```bash
cd ~/img2glb
bash setup.sh                 # venv 생성 + 설치, 10~20분
bash download_models.sh       # (선택) 가중치 사전 다운로드 (~13 GB → /data/sungju030307/hf-cache)
```

- 기준: RTX 3090 (24 GB), CUDA 12.4, Python 3.12, torch 2.6.0+cu124 (기존 venv 들과 동일)
- 경로는 전부 `env.sh` 에서 바꿀 수 있음 (`DATA_ROOT`)

## 2. 실행

```bash
source ~/img2glb/env.sh        # venv 활성화 + 캐시 경로 (매 셸마다 1번)
cd ~/img2glb

# 폴더 방식 (inputs/example/front.png ... 준비)
python generate.py --input_dir inputs/example --out outputs/example

# 파일 직접 지정
python generate.py --front f.png --back b.png --left l.png --right r.png --out outputs/char1

# 메시만 (빠름, 텍스처 X)
python generate.py --input_dir inputs/example --out outputs/example --no_texture

# 이미 만든 메시에 텍스처만 다시 입히기 (시드/뷰 바꿔 재시도할 때)
python generate.py --input_dir inputs/example --out outputs/example_v2 --mesh outputs/example/mesh_white.glb

# 여러 캐릭터를 GPU 4장에 나눠서 한 번에
bash batch.sh
```

결과:
- `outputs/<name>/mesh_white.glb`     형태만
- `outputs/<name>/mesh_textured.glb`  **텍스처 포함 GLB** ← Blender / three.js 에서 바로 열림
- `outputs/<name>/inputs/*.png`       실제 파이프라인에 들어간 (배경 제거된) 이미지

## 결과 보기 (Blender 불필요)

```bash
source ~/img2glb/env.sh
python ~/img2glb/viewer/serve.py          # 포트 8766
```
브라우저에서 `http://192.168.0.96:8766/viewer/` (같은 네트워크) 또는 VS Code 포트 포워딩으로 `http://localhost:8766/viewer/`.
`outputs/*/*.glb` 가 드롭다운에 뜨고, 로컬 PC 의 GLB 를 창에 드래그해서 열 수도 있음.
텍스처 포함 여부 / 삼각형 수가 상단에 표시됨. 최종 정리(색 보정, 리토폴로지)는 Blender 에서 `File > Import > glTF 2.0`.

## 3. 주요 옵션

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--steps` | 50 | shape diffusion step. 30 이면 빠르고 대충, 50 이상 권장 |
| `--octree` | 380 | 메시 해상도. 256(빠름) ~ 512(디테일, 느림/VRAM↑) |
| `--seed` | 12345 | 시드. 결과 마음에 안 들면 바꿔가며 재생성 |
| `--paint_subfolder` | `hunyuan3d-paint-v2-0-turbo` | `hunyuan3d-paint-v2-0` 으로 바꾸면 느리지만 품질↑ |
| `--paint_views` | `front,back,left,right` | 텍스처에 쓸 뷰. `front` 만 쓰면 공식 예제와 동일 |
| `--reduce_faces` | 0 | 예: `40000` → 텍스처 전에 면 수 축소 (게임/웹용) |
| `--bg` | `auto` | 배경 제거. `auto`: 단색 배경이면 색 키잉, 아니면 rembg. `key` / `rembg` 로 강제 |
| `--mesh` | - | 기존 GLB 를 주면 shape 건너뛰고 텍스처만 |
| `--gpu` | (env) | 쓸 GPU 번호. 안 주면 CUDA_VISIBLE_DEVICES 그대로 |
| `--low_vram` | off | shape 끝나면 shape 모델 내리고 paint 로드 |

## 4. 입력 이미지 팁 (SD 캐릭터 기준)

**가장 중요: 배경 제거.** 흰 배경에 흰/밝은 캐릭터를 rembg 에 맡기면 머리 같은 큰 밝은 영역을
통째로 지워버리고 알파도 반투명으로 만들어서, 메시는 속 빈 껍데기가 되고 텍스처는 뿌옇게 나옴 (sample1 에서 실제로 겪음).
그래서 `generate.py` 는 배경이 단색이면 rembg 대신 테두리 색 키잉(`--bg key`)을 자동으로 씀.
가장 안전한 건 처음부터 **투명 배경 PNG(RGBA)** 로 저장하는 것.

결과 폴더의 `inputs/*.png` 를 열어서 캐릭터가 온전히 남았는지 한 번 확인하는 습관을 들일 것.

- 4장 모두 **같은 캐릭터, 같은 포즈, 같은 크기**로. 각 뷰가 서로 안 맞으면 메시가 뭉개짐
- 투명 배경 PNG 가 가장 좋음. 배경 있으면 자동 rembg 하지만 머리카락 끝 등이 잘릴 수 있음
- 플랫 컬러 + 깔끔한 라인 스타일은 잘 따라오지만, 텍스처에 약간의 음영/얼룩이 생길 수 있음
  → 최종 얼굴/볼/리본 등은 Blender 에서 텍스처 살짝 정리하는 걸 전제로
- back 이 없으면 뒷면은 모델이 상상해서 만듦. 뒷면 디테일(리본 등) 있으면 꼭 넣기
- 정면 1장 또는 정면+뒷면만 넣으면 종이처럼 납작하게 나옴. **옆면(left/right) 이 부피를 만들어 줌** → 최소 정면+옆면
- `shade.py` 로 가짜 음영을 넣는 실험도 해봤는데 배경만 제대로 떼면 필요 없었음 (오히려 색이 탁해짐)
- **옆면 실루엣이 곧 두께다.** 리본·꼬리·귀처럼 튀어나온 부위는 옆면 그림에서 실제 두께대로 그려야 함.
  옆면에서 얇게 그리면 종이처럼 납작한 판이 됨 (sample1_character 머리 리본). 해상도·옵션으로는 못 고침
- 몸에 붙은 부위(앞발 등)는 윤곽선이나 톤으로 확실히 구분해 그려야 별개 덩어리로 나옴
- 뒷면에서 살짝 보이는 부위(리본 고리 끝 등)도 뒷면 그림에 그려 넣어야 텍스처가 검게 안 나옴 (`fix_bow.py` 로 임시 보정 가능)

## 5. 요구 사양

| | VRAM |
|---|---|
| shape 만 | ~6 GB |
| shape + texture | ~16 GB |

3090 한 장(24 GB) 으로 전체 파이프라인 OK. 4장이면 `batch.sh` 로 캐릭터 4개 동시 처리.

디스크: venv(~6 GB) + 가중치(shape fp16 ~3 GB, paint+delight ~10 GB) ≈ **20 GB**, 전부 `/data` 에 저장됨.

## 6. 트러블슈팅

- `trust_remote_code` 에러로 텍스처 단계 실패 → `bash patches/apply.sh` (diffusers 0.40 기준 필요. setup.sh 는 자동 적용)
- `cuda_runtime_api.h: No such file` 로 빌드 실패 → PATH 의 nvcc 가 anaconda 것. `env.sh` 가 `/usr/local/cuda` 로 고정하므로 `source env.sh && bash build_ext.sh`

- `custom_rasterizer` 빌드 실패 → `nvcc --version` 이 12.x 인지, `source env.sh` 후 `pip install ninja pybind11` 되어 있는지 확인
- OOM → `--octree 256` 또는 `--low_vram`, 또는 `--paint_subfolder hunyuan3d-paint-v2-0-turbo`
- 텍스처가 뿌옇게 나옴 → `--paint_subfolder hunyuan3d-paint-v2-0`, 입력 이미지 해상도 1024 이상으로
- 뷰가 엉킴 (꼬리 위치가 뷰마다 다름 등) → left/right 가 바뀐 것. **규칙: `left.png` 는 캐릭터가 화면 왼쪽을 보는 그림**
  (카메라가 캐릭터의 왼쪽에 있음). 반대로 그려졌으면 파일을 안 바꾸고 `--left x/right.png --right x/left.png` 로 교체 지정 가능
