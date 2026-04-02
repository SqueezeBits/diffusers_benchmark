# diffusers_benchmark

`diffusers_benchmark` is a small benchmark repository for measuring stage-level latency in FLUX pipelines built with Hugging Face `diffusers`.

The current script measures the following stages:

- `prompt_embedding`
- `vae_encode`
- `denoising_step`
- `decode_latent`

Supported modes:

- `t2i` (text-to-image)
- `i2i` (image-to-image)

## Supported Models

The script currently supports the following FLUX model families based on the Hugging Face model ID:

| Model family pattern | Mode | Diffusers pipeline | Example model ID |
| --- | --- | --- | --- |
| `FLUX.1*` | `t2i` | `FluxPipeline` | `black-forest-labs/FLUX.1-dev` |
| `FLUX.1*` | `i2i` | `FluxImg2ImgPipeline` | `black-forest-labs/FLUX.1-dev` |
| `FLUX.2*` | `t2i` | `Flux2Pipeline` | `black-forest-labs/FLUX.2-dev` |
| `FLUX.2*` | `i2i` | `Flux2Pipeline` | `black-forest-labs/FLUX.2-dev` |
| `FLUX.2-Klein*` | `t2i` | `Flux2KleinPipeline` | `black-forest-labs/FLUX.2-Klein` |
| `FLUX.2-Klein*` | `i2i` | `Flux2KleinPipeline` | `black-forest-labs/FLUX.2-Klein` |
| `Wan-Animate*` | `v2v` | `WanAnimatePipeline` | `Wan-AI/Wan2.2-Animate-14B-Diffusers` |

Model support is inferred from the model ID string. If the ID does not contain `flux.1`, `flux.2`, `flux.2-klein`, or both `wan` and `animate`, the script will reject it as unsupported.


## Requirements

- Linux
- One NVIDIA GPU
- CUDA 13.0 or newer
- Python 3.13
- Hugging Face access permission and login token if you want to use gated models such as `black-forest-labs/FLUX.2-dev`

## Dependency Management With `uv`

This repository uses `uv` to manage all Python dependencies, including `torch` and `torchvision`.

Because this project is intended to run only on CUDA 13.0+ environments, `torch` is included directly in the default dependency set.

Create the environment and install dependencies with:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync
```

If you already have an active environment and only want to add the project dependencies:

```bash
uv sync --active
```


## Usage

### FLUX.2-dev text-to-image benchmark

The command below is the same benchmark flow you requested, with shell quoting fixed for Bash. Because the prompt itself contains double quotes such as `"WATER"`, it is safer to wrap the full prompt in single quotes.

```bash
uv run python benchmark.py \
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
uv run python benchmark.py \
  --model "black-forest-labs/FLUX.2-dev" \
  --mode i2i \
  --image input.png \
  --prompt "A cinematic watercolor reinterpretation" \
  --num-inference-steps 28 \
  --warmup 1 \
  --iterations 5 \
  --output outputs/i2i-output.png \
  --save-json outputs/i2i-bench.json
```

### Wan-Animate benchmark

`WanAnimatePipeline` requires a character reference image, a pose video, and a face video. The `--mode` flag is ignored for Wan-Animate models.

**Move mode** (default):

```bash
uv run python benchmark.py \
  --model "Wan-AI/Wan2.2-Animate-14B-Diffusers" \
  --image wan_test_assets/character.jpeg \
  --pose-video wan_test_assets/pose_official.mp4 \
  --face-video wan_test_assets/face_official.mp4 \
  --prompt "A character moving naturally." \
  --num-inference-steps 20 \
  --num-frames 77 \
  --height 480 \
  --width 848 \
  --guidance-scale 1 \
  --warmup 3 \
  --iterations 5 \
  --output outputs/wan_animate_output.mp4 \
  --save-json outputs/wan_animate.json
```

**Replace mode** (default):

```bash
uv run python benchmark.py \
  --model "Wan-AI/Wan2.2-Animate-14B-Diffusers" \
  --image wan_test_assets/character.jpeg \
  --pose-video wan_test_assets/pose_official.mp4 \
  --face-video wan_test_assets/face_official.mp4 \
  --prompt "A character moving naturally." \
  --num-inference-steps 20 \
  --num-frames 77 \
  --height 480 \
  --width 848 \
  --guidance-scale 1 \
  --warmup 3 \
  --iterations 5 \
  --output outputs/wan_animate_replace_output.mp4 \
  --save-json outputs/wan_animate_replace.json \
  --wan-mode replace \
  --background-video wan_test_assets/bg_from_motion_bbox_medium.mp4 \
  --mask-video wan_test_assets/mask_from_motion_bbox_medium.mp4
```

## Main Options

- `--model`: Hugging Face model ID
- `--mode`: `t2i` or `i2i`
- `--prompt`: input prompt
- `--image`: input image; required in `i2i` mode and always required for Wan-Animate (character reference)
- `--num-inference-steps`: number of denoising steps
- `--warmup`: number of warmup runs before measurement
- `--iterations`: number of measured iterations
- `--output`: output path for the last measured image or video
- `--save-json`: path for aggregate and per-iteration benchmark results
- `--disable-compile`: disable `torch.compile`
- `--compile-mode`: set the `torch.compile` mode

### Wan-Animate options

- `--pose-video`: pose video path (required for Wan-Animate)
- `--face-video`: face video path (required for Wan-Animate)
- `--background-video`: background video path (required for `replace` mode)
- `--mask-video`: mask video path (required for `replace` mode)
- `--num-frames`: `segment_frame_length` passed to the pipeline (default: `77`)
- `--wan-mode`: pipeline mode — `animate` (default) or `replace`
- `--output-fps`: frames per second when saving video output (default: `30`)

## Output

After execution, the script prints per-stage mean and max latency to the terminal and can optionally generate:

- a PNG image (Image Generation) or an MP4 video (Video Generation)
- a JSON benchmark report

## Notes

- The script currently assumes a single device at `cuda:0`.
- It fails immediately if CUDA is not available.
- `--image` is required in `i2i` mode and always required for Wan-Animate (character reference image).
- `--pose-video` and `--face-video` are required for all Wan-Animate runs.
- `--background-video` and `--mask-video` are additionally required when `--wan-mode replace` is used.
- If `torch.compile` is slow or unstable in your environment, use `--disable-compile`.
