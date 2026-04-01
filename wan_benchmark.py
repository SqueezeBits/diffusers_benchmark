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
    components: dict[str, tuple[int, float]] = field(default_factory=dict)


class StageTimer:
    """Measures per-component timings with CUDA synchronization."""

    def __init__(self) -> None:
        self.samples: dict[str, list[float]] = {}

    def reset(self) -> None:
        self.samples.clear()

    def measure(self, name: str):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                yield
            finally:
                torch.cuda.synchronize()
                self.samples.setdefault(name, []).append(
                    (time.perf_counter() - t0) * 1000
                )

        return _ctx()

    def summary(self) -> dict[str, tuple[int, float]]:
        """Returns {name: (calls, total_ms)}."""
        return {k: (len(v), sum(v)) for k, v in self.samples.items()}


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

        log.info("torch.compile (dynamic=True)...")
        # Compile transformer block-by-block (like MAX) for fair comparison
        # Use no-cudagraphs to avoid CUDAGraph overwrite errors with CFG
        if compile_mode == "layer-wise":
            for i, block in enumerate(pipe.transformer.blocks):
                pipe.transformer.blocks[i] = torch.compile(
                    block, mode="max-autotune-no-cudagraphs", dynamic=True
                )
        else:
            pipe.transformer = torch.compile(
                pipe.transformer, mode="max-autotune-no-cudagraphs", dynamic=True
            )
        pipe.vae.encode = torch.compile(
            pipe.vae.encode, mode=compile_mode, dynamic=True
        )
        pipe.vae.decode = torch.compile(
            pipe.vae.decode, mode=compile_mode, dynamic=True
        )
        pipe.text_encoder = torch.compile(
            pipe.text_encoder, dynamic=True
        )

        # Instrument pipeline for component-level timing
        timer = StageTimer()
        _orig_encode_prompt = pipe.encode_prompt
        _orig_transformer_fwd = pipe.transformer.forward
        _orig_vae_decode = pipe.vae.decode
        _orig_vae_encode = getattr(pipe.vae, "encode", None)

        def _timed_encode_prompt(*a, **kw):
            with timer.measure("text_encoder"):
                return _orig_encode_prompt(*a, **kw)

        def _timed_transformer(*a, **kw):
            with timer.measure("transformer"):
                return _orig_transformer_fwd(*a, **kw)

        def _timed_vae_decode(*a, **kw):
            with timer.measure("vae_decode"):
                return _orig_vae_decode(*a, **kw)

        def _timed_vae_encode(*a, **kw):
            with timer.measure("vae_encode"):
                return _orig_vae_encode(*a, **kw)

        pipe.encode_prompt = _timed_encode_prompt
        pipe.transformer.forward = _timed_transformer
        pipe.vae.decode = _timed_vae_decode
        if _orig_vae_encode is not None:
            pipe.vae.encode = _timed_vae_encode

        # Warmup with small resolution to trigger torch.compile
        log.info("Warmup (480x832, 17 frames, 2 steps)...")
        warmup_kwargs: dict = {
            "prompt": prompt,
            "negative_prompt": "low quality",
            "height": 480,
            "width": 832,
            "num_frames": 17,
            "guidance_scale": cfg["guidance"],
            "num_inference_steps": 2,
        }
        if cfg["guidance_2"] is not None:
            warmup_kwargs["guidance_scale_2"] = cfg["guidance_2"]
        if cfg["mode"] == "i2v":
            warmup_kwargs["image"] = load_i2v_image(480, 832)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipe(**warmup_kwargs).frames[0]
        torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - t0) * 1000
        log.info("Warmup done: %.0fms", warmup_ms)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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

            # Timed run
            log.info("(%d/%d) %s — timed run", idx, total, tag)
            timer.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pipe_kwargs["output_type"] = "np"
            output = pipe(**pipe_kwargs).frames[0]
            torch.cuda.synchronize()
            e2e_ms = (time.perf_counter() - t0) * 1000
            components = timer.summary()

            # Save video (h264 via av for compatibility)
            video_path = output_dir / f"{model_name}_{label}.mp4"
            import av

            with av.open(str(video_path), mode="w") as container:
                stream = container.add_stream("h264", rate=16)
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                for frame_np in output:
                    frame_u8 = (np.clip(frame_np, 0, 1) * 255).astype(np.uint8)
                    frame = av.VideoFrame.from_ndarray(frame_u8, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            log.info(
                "(%d/%d) %s — E2E %.0fms (warmup %.0fms) → %s",
                idx, total, tag, e2e_ms, warmup_ms, video_path,
            )
            for comp, (calls, ms) in sorted(components.items()):
                log.info("  %s: %.0fms (%d calls, %.1fms avg)", comp, ms, calls, ms / calls if calls else 0)
            results.append(
                TimingResult(model_name, label, e2e_ms, warmup_ms, components)
            )

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

    model_names = list(dict.fromkeys(r.model for r in results))

    for model_name in model_names:
        model_results = [r for r in results if r.model == model_name]
        for r in model_results:
            # Collect all component names across results
            comp_names = sorted(
                {k for mr in model_results for k in mr.components}
            )

            print(f"\n==================== PROFILING REPORT ==")
            print(f"  {gpu} | {model_name} | {r.label} | {steps} steps")
            print(f"  torch.compile (max-autotune-no-cudagraphs, dynamic=True)")

            # Component timings (calls from timer samples)
            print(f"\nComponent Timings:")
            hdr = f"{'components':<30} {'calls':>7} {'total':>12} {'avg':>12} (ms)"
            print(hdr)
            for comp in comp_names:
                calls, ms = r.components.get(comp, (0, 0))
                avg = ms / calls if calls else 0
                print(f"{comp:<30} {calls:>7} {ms:>12.3f} {avg:>12.3f}")

            # Method timings
            print(f"\nMethod Timings:")
            hdr = f"{'methods':<30} {'calls':>7} {'total':>12} {'avg':>12} (ms)"
            print(hdr)
            print(f"{'E2E execute':<30} {'1':>7} {r.e2e_ms:>12.3f} {r.e2e_ms:>12.3f}")
            for comp in comp_names:
                calls, ms = r.components.get(comp, (0, 0))
                avg = ms / calls if calls else 0
                print(f"{comp:<30} {calls:>7} {ms:>12.3f} {avg:>12.3f}")
            print(f"{'warmup':<30} {'1':>7} {r.warmup_ms:>12.3f} {r.warmup_ms:>12.3f}")
            print(f"==========================================")

    # Summary table
    print(f"\n{'=' * 65}")
    print(f"  Summary — {gpu}, {steps} steps")
    print(f"{'=' * 65}\n")

    hdr = f"{'Model':<22} {'Resolution':<12} {'E2E (ms)':>10}"
    print(hdr)
    print("-" * len(hdr))

    for model_name in model_names:
        model_results = [r for r in results if r.model == model_name]
        for r in model_results:
            print(f"{r.model:<22} {r.label:<12} {r.e2e_ms:>10.0f}")
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
