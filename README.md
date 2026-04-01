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
| `FLUX.2-klein*fp8*` | `t2i` | `Flux2KleinPipeline` + fp8 DiT patch | `black-forest-labs/FLUX.2-klein-4b-fp8` |
| `FLUX.2-klein*fp8*` | `i2i` | `Flux2KleinPipeline` + fp8 DiT patch | `black-forest-labs/FLUX.2-klein-4b-fp8` |

Model support is inferred from the model ID string. If the ID does not contain `flux.1`, `flux.2`, or `flux.2-klein`, the script will reject it as unsupported.
For `FLUX.2-klein` fp8 checkpoints, the benchmark loads the matching bf16 base pipeline first, then swaps only the DiT linear layers to fp8. The text encoder and VAE remain bf16.


## Requirements

- Linux
- One NVIDIA GPU
- CUDA 13.0 or newer
- Python 3.12 or 3.13
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

If you see a `torch` import failure on Python 3.13 in your environment, recreate the venv with Python 3.12 instead:

```bash
uv venv --python 3.12
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

### FLUX.2-klein fp8 benchmark

```bash
uv run python benchmark.py \
  --model "black-forest-labs/FLUX.2-klein-4b-fp8" \
  --mode t2i \
  --prompt "A cat in a garden" \
  --num-inference-steps 28 \
  --warmup 1 \
  --iterations 5 \
  --output outputs/flux2-klein-fp8-output.png \
  --save-json outputs/flux2-klein-fp8-bench.json
```

The official `FLUX.2-klein-4b-fp8` repo contains only DiT weights, so the benchmark automatically uses `black-forest-labs/FLUX.2-klein-4B` as the bf16 base pipeline. If you use a different DiT-only fp8 repo, pass the matching full pipeline with `--base-model`.

## Main Options

- `--model`: Hugging Face model ID
- `--base-model`: full pipeline model ID used when `--model` is a DiT-only checkpoint
- `--mode`: `t2i` or `i2i`
- `--prompt`: input prompt
- `--image`: input image used in `i2i` mode
- `--num-inference-steps`: number of denoising steps
- `--warmup`: number of warmup runs before measurement
- `--iterations`: number of measured iterations
- `--output`: output path for the last measured image
- `--save-json`: path for aggregate and per-iteration benchmark results
- `--disable-compile`: disable `torch.compile`
- `--compile-mode`: set the `torch.compile` mode

## Output

After execution, the script prints per-stage mean and max latency to the terminal and can optionally generate:

- a PNG image
- a JSON benchmark report

## Notes

- The script currently assumes a single device at `cuda:0`.
- It fails immediately if CUDA is not available.
- `--image` is required in `i2i` mode.
- If `torch.compile` is slow or unstable in your environment, use `--disable-compile`.
