# diffusers_benchmark

`diffusers_benchmark`는 Hugging Face `diffusers` 기반 FLUX 파이프라인에서 주요 단계별 시간을 측정하는 간단한 벤치마크 저장소입니다.

현재 스크립트는 다음 단계를 측정합니다.

- `prompt_embedding`
- `vae_encode`
- `denoising_step`
- `decode_latent`

지원하는 모드는 다음과 같습니다.

- `t2i` (text-to-image)
- `i2i` (image-to-image)

## Repository Layout

- `benchmark_flux_stage_times.py`: 메인 벤치마크 스크립트
- `pyproject.toml`: `uv` 기반 의존성 정의
- `.gitignore`: 가상환경 및 산출물 제외 규칙

## Requirements

- Linux
- NVIDIA GPU 1개
- CUDA가 정상 동작하는 PyTorch 환경
- Python 3.12 또는 3.13
- `black-forest-labs/FLUX.2-dev` 같은 gated 모델을 사용할 경우 Hugging Face 접근 권한과 로그인 토큰

## Dependency Management With `uv`

이 저장소는 공통 Python 의존성을 `uv`로 관리합니다.

`torch`는 CUDA 버전과 드라이버 환경에 따라 설치 방식이 달라지므로 optional dependency로 분리했습니다. 따라서 아래 둘 중 하나를 선택하면 됩니다.

### Option A: generic PyTorch 설치까지 `uv`에 맡기기

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra torch
```

### Option B: CUDA에 맞는 PyTorch를 먼저 설치한 뒤 나머지만 동기화하기

이 방식이 GPU 서버에서는 더 안전합니다.

```bash
uv venv --python 3.13
source .venv/bin/activate

# 예시: CUDA 환경에 맞는 torch를 먼저 설치
uv pip install torch

# pyproject.toml에 정의된 나머지 의존성 설치
uv sync
```

이미 활성화된 CUDA/PyTorch 환경에 의존성만 추가하려면:

```bash
uv sync --active
```

## Hugging Face Login

gated 모델을 쓰는 경우 먼저 Hugging Face 로그인이 필요할 수 있습니다.

```bash
huggingface-cli login
```

또는 `HF_TOKEN` 환경 변수를 사용해도 됩니다.

## Usage

### FLUX.2-dev text-to-image benchmark

아래 예시는 질문에 포함된 명령을 bash에서 안전하게 실행되도록 quoting만 수정한 버전입니다. 프롬프트 안에 `"WATER"` 같은 큰따옴표가 들어가므로, 전체 프롬프트는 작은따옴표로 감싸는 편이 안전합니다.

```bash
uv run python benchmark_flux_stage_times.py \
  --model "black-forest-labs/FLUX.2-dev" \
  --mode t2i \
  --prompt 'Four elements split into quadrants: top-left shows splashing water forming the word "WATER" on a water background, top-right shows soil forming "EARTH" with planet earth behind, bottom-left shows colorful clouds forming "AIR" at sunset, bottom-right shows fiery lava forming "FIRE" against the sun' \
  --num-inference-steps 50 \
  --warmup 3 \
  --iterations 5 \
  --output outputs/flux2-dev-output.png \
  --save-json outputs/flux2-dev-bench.json
```

### Image-to-image benchmark

```bash
uv run python benchmark_flux_stage_times.py \
  --model "black-forest-labs/FLUX.1-dev" \
  --mode i2i \
  --image input.png \
  --prompt "A cinematic watercolor reinterpretation" \
  --num-inference-steps 28 \
  --warmup 1 \
  --iterations 5 \
  --output outputs/i2i-output.png \
  --save-json outputs/i2i-bench.json
```

## Main Options

- `--model`: Hugging Face model id
- `--mode`: `t2i` 또는 `i2i`
- `--prompt`: 입력 프롬프트
- `--image`: `i2i` 모드에서 사용할 입력 이미지
- `--num-inference-steps`: 디노이징 step 수
- `--warmup`: 측정 전 워밍업 횟수
- `--iterations`: 실제 측정 횟수
- `--output`: 마지막 measured iteration의 이미지 저장 경로
- `--save-json`: aggregate + per-iteration 결과 저장 경로
- `--disable-compile`: `torch.compile` 비활성화
- `--compile-mode`: `torch.compile` mode 지정

## Output

실행이 끝나면 터미널에는 단계별 평균/최대 시간이 출력되고, 선택적으로 아래 파일이 생성됩니다.

- PNG 이미지
- JSON 벤치마크 결과

`outputs/` 디렉터리는 `.gitignore`에 포함되어 있으므로 벤치마크 산출물이 기본적으로 git에 섞이지 않습니다.

## Notes

- 스크립트는 현재 `cuda:0` 하나를 전제로 동작합니다.
- CUDA를 사용할 수 없으면 즉시 실패합니다.
- `i2i` 모드에서는 `--image`가 필수입니다.
- `torch.compile`이 환경에 따라 느리거나 불안정하면 `--disable-compile` 옵션을 사용하면 됩니다.

## Git

로컬 저장소 초기화 후 아래처럼 커밋하면 됩니다.

```bash
git init
git add .
git commit -m "Initial diffusers benchmark setup"
```

원격 저장소가 있다면 이후에 `git remote add origin ...` 및 `git push -u origin main`으로 올리면 됩니다.
