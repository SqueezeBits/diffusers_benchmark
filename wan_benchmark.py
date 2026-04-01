#!/usr/bin/env python3
"""Wan video generation benchmark using diffusers + torch.compile.

Usage:
    uv run wan_benchmark.py
    uv run wan_benchmark.py --model wan2.2-t2v-a14b
    uv run wan_benchmark.py --model all --steps 40
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wan_benchmark")

I2V_IMAGE_URL = (
    "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B"
    "/resolve/main/examples/i2v_input.JPG"
)

MODELS = {
    "wan2.2-t2v-a14b": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "cls": "WanPipeline",
        "mode": "t2v",
        "guidance": 4.0,
        "guidance_2": 3.0,
        "prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright "
            "gloves fight intensely on a spotlighted stage."
        ),
    },
    "wan2.2-i2v-a14b": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "cls": "WanImageToVideoPipeline",
        "mode": "i2v",
        "guidance": 4.0,
        "guidance_2": 3.0,
        "prompt": "A cat surfing on a wave",
    },
    "wan2.1-t2v-14b": {
        "repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "cls": "WanPipeline",
        "mode": "t2v",
        "guidance": 5.0,
        "guidance_2": None,
        "prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright "
            "gloves fight intensely on a spotlighted stage."
        ),
    },
    "wan2.1-i2v-14b": {
        "repo": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "cls": "WanImageToVideoPipeline",
        "mode": "i2v",
        "guidance": 5.0,
        "guidance_2": None,
        "prompt": "A cat surfing on a wave",
    },
}

RESOLUTIONS = [
    {"height": 720, "width": 1280, "num_frames": 81, "label": "1280x720"},
    {"height": 1280, "width": 720, "num_frames": 81, "label": "720x1280"},
]


@dataclass
class TimingResult:
    model: str
    label: str
    e2e_ms: float
    warmup_ms: float = 0.0


def load_i2v_image(height: int, width: int) -> Image.Image:
    import io
    import urllib.request

    log.info("Downloading I2V input image...")
    with urllib.request.urlopen(I2V_IMAGE_URL) as resp:
        img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    return img.resize((width, height))


def patch_umt5() -> None:
    """Workaround: transformers UMT5 tied-embedding bug.
    https://github.com/huggingface/transformers/issues/43992
    """
    from transformers import UMT5EncoderModel

    _orig = UMT5EncoderModel.__init__

    def _patched(self, config):
        _orig(self, config)
        self.encoder.embed_tokens = self.shared

    UMT5EncoderModel.__init__ = _patched


def run_benchmark(
    model_names: list[str],
    steps: int,
    output_dir: Path,
    compile_mode: str,
) -> list[TimingResult]:
    import diffusers

    patch_umt5()

    results: list[TimingResult] = []
    total = len(model_names) * len(RESOLUTIONS)
    idx = 0

    for model_name in model_names:
        cfg = MODELS[model_name]
        pipe_cls = getattr(diffusers, cfg["cls"])
        prompt = cfg["prompt"]

        log.info("Loading %s (%s)...", model_name, cfg["repo"])
        pipe = pipe_cls.from_pretrained(
            cfg["repo"], torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")

        log.info("torch.compile (mode=%s, dynamic=True)...", compile_mode)
        pipe.transformer = torch.compile(
            pipe.transformer, mode=compile_mode, dynamic=True
        )
        pipe.vae = torch.compile(
            pipe.vae, mode=compile_mode, dynamic=True
        )
        pipe.text_encoder = torch.compile(
            pipe.text_encoder, dynamic=True
        )

        for res in RESOLUTIONS:
            idx += 1
            label = str(res["label"])
            height, width = int(res["height"]), int(res["width"])
            num_frames = int(res["num_frames"])
            tag = f"{model_name}/{label}"
            log.info("(%d/%d) %s", idx, total, tag)

            pipe_kwargs: dict = {
                "prompt": prompt,
                "negative_prompt": "low quality",
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "guidance_scale": cfg["guidance"],
                "num_inference_steps": steps,
            }
            if cfg["guidance_2"] is not None:
                pipe_kwargs["guidance_scale_2"] = cfg["guidance_2"]
            if cfg["mode"] == "i2v":
                pipe_kwargs["image"] = load_i2v_image(height, width)

            # Warmup
            log.info("(%d/%d) %s — warmup", idx, total, tag)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = pipe(**pipe_kwargs).frames[0]
            torch.cuda.synchronize()
            warmup_ms = (time.perf_counter() - t0) * 1000
            log.info("(%d/%d) %s — warmup %.0fms", idx, total, tag, warmup_ms)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Timed run
            log.info("(%d/%d) %s — timed run", idx, total, tag)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipe_kwargs["output_type"] = "np"
            output = pipe(**pipe_kwargs).frames[0]
            torch.cuda.synchronize()
            e2e_ms = (time.perf_counter() - t0) * 1000

            # Save video
            video_path = output_dir / f"{model_name}_{label}.mp4"
            from diffusers.utils import export_to_video

            export_to_video(output, str(video_path), fps=16)
            log.info(
                "(%d/%d) %s — E2E %.0fms (warmup %.0fms) → %s",
                idx, total, tag, e2e_ms, warmup_ms, video_path,
            )
            results.append(TimingResult(model_name, label, e2e_ms, warmup_ms))

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    return results


def print_summary(results: list[TimingResult], steps: int) -> None:
    import subprocess

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        gpu = "unknown"

    print(f"\n{'=' * 65}")
    print(f"  Diffusers Wan Benchmark — {gpu}")
    print(f"  {steps} steps, torch.compile (max-autotune, dynamic=True)")
    print(f"{'=' * 65}\n")

    hdr = f"{'Model':<22} {'Resolution':<12} {'E2E (ms)':>10} {'Warmup':>10}"
    print(hdr)
    print("-" * len(hdr))

    model_names = list(dict.fromkeys(r.model for r in results))
    for model_name in model_names:
        model_results = [r for r in results if r.model == model_name]
        for r in model_results:
            print(f"{r.model:<22} {r.label:<12} {r.e2e_ms:>10.0f} {r.warmup_ms:>10.0f}")
        if len(model_results) > 1:
            avg = mean(r.e2e_ms for r in model_results)
            print(f"{model_name:<22} {'avg':<12} {avg:>10.0f}")
        print()

    # Save JSON
    json_data = [asdict(r) for r in results]
    print(json.dumps(json_data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wan video generation benchmark (diffusers)"
    )
    parser.add_argument(
        "--model", nargs="*", default=["wan2.2-t2v-a14b"],
        help=f"Choices: {', '.join(MODELS)}, all",
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output-dir", default="/tmp/diffusers_wan_benchmark")
    parser.add_argument("--compile-mode", default="max-autotune")
    args = parser.parse_args()

    names = list(MODELS.keys()) if "all" in args.model else args.model
    for m in names:
        if m not in MODELS:
            print(f"Unknown: {m}. Choices: {', '.join(MODELS)}")
            sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(names, args.steps, out, args.compile_mode)
    print_summary(results, args.steps)


if __name__ == "__main__":
    main()
