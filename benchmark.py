#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import inspect
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from PIL import Image

from diffusers import Flux2KleinPipeline, Flux2Pipeline, FluxImg2ImgPipeline, FluxPipeline


@dataclass
class IterationMetrics:
    total_ms: float
    prompt_embedding_ms: float
    vae_encode_ms: float
    vae_encode_calls: int
    denoising_step_ms: float
    denoising_total_ms: float
    denoising_calls: int
    decode_latent_ms: float


class StageTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.samples: dict[str, list[float]] = defaultdict(list)
        self._span_starts: dict[str, float] = {}

    def reset(self) -> None:
        self.samples.clear()
        self._span_starts.clear()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name: str):
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.samples[name].append((time.perf_counter() - start) * 1000.0)

    def start_span(self, name: str) -> None:
        self.synchronize()
        self._span_starts[name] = time.perf_counter()

    def end_span(self, name: str) -> None:
        start = self._span_starts.pop(name, None)
        if start is None:
            return
        self.synchronize()
        self.samples[name].append((time.perf_counter() - start) * 1000.0)


class MethodPatch:
    def __init__(self, obj: Any, name: str, wrapped: Any) -> None:
        self.obj = obj
        self.name = name
        self.original = getattr(obj, name)
        setattr(obj, name, wrapped)

    def restore(self) -> None:
        setattr(self.obj, self.name, self.original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark prompt embedding, denoising step, and latent decode for FLUX diffusers pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--mode", choices=("t2i", "i2i"), default="t2i", help="Benchmark mode.")
    parser.add_argument("--prompt", default="A cat in a garden", help="Prompt text.")
    parser.add_argument("--image", help="Init image path for i2i mode.")
    parser.add_argument("--height", type=int, default=1024, help="Output height.")
    parser.add_argument("--width", type=int, default=1024, help="Output width.")
    parser.add_argument("--strength", type=float, default=0.6, help="Img2Img strength when supported by the pipeline.")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="Guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, default=28, help="Number of denoising steps.")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations run before measurement.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", help="Optional path to save the first image from the last measured iteration.")
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile. By default text encoder, DiT, and VAE are compiled.",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune",
        help="torch.compile mode passed to compiled components.",
    )
    parser.add_argument(
        "--save-json",
        help="Optional path to save aggregate and per-iteration benchmark results.",
    )
    args = parser.parse_args()
    if args.mode == "i2i" and not args.image:
        parser.error("--image is required when --mode i2i is selected.")
    return args


def infer_model_family(model_id: str) -> str:
    model_id_lower = model_id.lower()
    if "flux.2-klein" in model_id_lower:
        return "flux2_klein"
    if "flux.2" in model_id_lower:
        return "flux2"
    if "flux.1" in model_id_lower:
        return "flux1"
    raise ValueError(f"Unsupported model family for: {model_id}")


def load_pipeline(
    model_id: str,
    mode: str,
    device: torch.device,
):
    family = infer_model_family(model_id)
    pipeline_cls = {
        "flux1": FluxPipeline if mode == "t2i" else FluxImg2ImgPipeline,
        "flux2": Flux2Pipeline,
        "flux2_klein": Flux2KleinPipeline,
    }[family]

    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to(device=device, dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe._benchmark_pipeline_dtype = torch.bfloat16
    return pipe


def load_input_image(path: str, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((width, height))


def apply_compile(
    pipe: Any,
    enabled: bool,
    compile_mode: str,
) -> None:
    if not enabled:
        return

    def compile_method(fn: Any) -> Any:
        return torch.compile(fn, mode=compile_mode, fullgraph=False)

    for attr_name in ("text_encoder", "text_encoder_2"):
        module = getattr(pipe, attr_name, None)
        if module is not None:
            print(f"Compiling {attr_name}...")
            module.forward = compile_method(module.forward)

    if getattr(pipe, "transformer", None) is not None:
        print("Compiling transformer...")
        pipe.transformer.forward = compile_method(pipe.transformer.forward)

    if getattr(pipe, "vae", None) is not None:
        if hasattr(pipe.vae, "encode"):
            print("Compiling vae.encode...")
            pipe.vae.encode = compile_method(pipe.vae.encode)
        if hasattr(pipe.vae, "decode"):
            print("Compiling vae.decode...")
            pipe.vae.decode = compile_method(pipe.vae.decode)


@contextmanager
def instrument_pipeline(pipe: Any, stage_timer: StageTimer):
    patches: list[MethodPatch] = []

    def wrap_stage(name: str, fn: Any):
        def wrapped(*args, **kwargs):
            with stage_timer.measure(name):
                return fn(*args, **kwargs)

        return wrapped

    def wrap_decode_start(fn: Any):
        def wrapped(*args, **kwargs):
            stage_timer.start_span("decode_latent")
            return fn(*args, **kwargs)

        return wrapped

    def wrap_decode_end(fn: Any):
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                stage_timer.end_span("decode_latent")

        return wrapped

    patches.append(MethodPatch(pipe, "encode_prompt", wrap_stage("prompt_embedding", pipe.encode_prompt)))
    patches.append(
        MethodPatch(pipe.transformer, "forward", wrap_stage("denoising_step", pipe.transformer.forward))
    )
    if hasattr(pipe.vae, "encode"):
        patches.append(MethodPatch(pipe.vae, "encode", wrap_stage("vae_encode", pipe.vae.encode)))

    decode_start_method = None
    if hasattr(pipe, "_unpack_latents_with_ids"):
        decode_start_method = "_unpack_latents_with_ids"
    elif hasattr(pipe, "_unpack_latents"):
        decode_start_method = "_unpack_latents"

    if decode_start_method is not None:
        decode_start_fn = getattr(pipe, decode_start_method)
        patches.append(MethodPatch(pipe, decode_start_method, wrap_decode_start(decode_start_fn)))
        patches.append(
            MethodPatch(
                pipe.image_processor,
                "postprocess",
                wrap_decode_end(pipe.image_processor.postprocess),
            )
        )

    try:
        yield
    finally:
        for patch in reversed(patches):
            patch.restore()


def build_call_kwargs(
    pipe: Any,
    args: argparse.Namespace,
    init_image: Image.Image | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "return_dict": False,
    }

    if args.mode == "i2i":
        signature = inspect.signature(pipe.__call__)
        kwargs["image"] = init_image
        if "strength" in signature.parameters:
            kwargs["strength"] = args.strength

    return kwargs


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def collect_iteration_metrics(stage_timer: StageTimer, total_ms: float) -> IterationMetrics:
    prompt_embedding_ms = sum(stage_timer.samples.get("prompt_embedding", []))
    vae_encode_samples = stage_timer.samples.get("vae_encode", [])
    denoising_samples = stage_timer.samples.get("denoising_step", [])
    decode_latent_ms = sum(stage_timer.samples.get("decode_latent", []))

    vae_encode_ms = sum(vae_encode_samples)
    vae_encode_calls = len(vae_encode_samples)
    denoising_total_ms = sum(denoising_samples)
    denoising_calls = len(denoising_samples)
    denoising_step_ms = denoising_total_ms / denoising_calls if denoising_calls else 0.0

    return IterationMetrics(
        total_ms=total_ms,
        prompt_embedding_ms=prompt_embedding_ms,
        vae_encode_ms=vae_encode_ms,
        vae_encode_calls=vae_encode_calls,
        denoising_step_ms=denoising_step_ms,
        denoising_total_ms=denoising_total_ms,
        denoising_calls=denoising_calls,
        decode_latent_ms=decode_latent_ms,
    )


def ensure_parent_dir(path: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def extract_first_image(output: Any) -> Image.Image | np.ndarray:
    image = output
    if isinstance(image, tuple):
        image = image[0]
    if isinstance(image, list):
        if not image:
            raise ValueError("Pipeline returned an empty image list.")
        image = image[0]
    return image


def save_output_image(output: Any, path: str) -> None:
    image = extract_first_image(output)
    output_path = ensure_parent_dir(path)

    if isinstance(image, Image.Image):
        image.save(output_path)
        return

    if isinstance(image, np.ndarray):
        if image.ndim == 4:
            image = image[0]
        Image.fromarray(image).save(output_path)
        return

    raise TypeError(f"Unsupported image output type: {type(image)}")


def run_once(
    pipe: Any,
    base_call_kwargs: dict[str, Any],
    stage_timer: StageTimer,
    device: torch.device,
    seed: int,
) -> tuple[IterationMetrics, Any]:
    call_kwargs = dict(base_call_kwargs)
    call_kwargs["generator"] = make_generator(device=device, seed=seed)
    stage_timer.reset()
    stage_timer.synchronize()
    start = time.perf_counter()
    output = pipe(**call_kwargs)
    stage_timer.synchronize()
    total_ms = (time.perf_counter() - start) * 1000.0
    return collect_iteration_metrics(stage_timer=stage_timer, total_ms=total_ms), output


def summarize(values: list[float], *, count_override: int | None = None) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "max_ms": 0.0}
    return {
        "count": count_override if count_override is not None else len(values),
        "mean_ms": mean(values),
        "max_ms": max(values),
    }


def print_summary(results: list[IterationMetrics]) -> None:
    total_ms = [result.total_ms for result in results]
    prompt_embedding_ms = [result.prompt_embedding_ms for result in results]
    vae_encode_ms = [result.vae_encode_ms for result in results if result.vae_encode_calls > 0]
    denoising_step_ms = [result.denoising_step_ms for result in results]
    decode_latent_ms = [result.decode_latent_ms for result in results]
    vae_encode_count = sum(result.vae_encode_calls for result in results)
    denoising_step_count = sum(result.denoising_calls for result in results)

    print("\nBenchmark results (ms)")
    print(f"{'stage':<20} {'count':>8} {'mean':>12} {'max':>12}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 12} {'-' * 12}")
    for name, values, count_override in (
        ("total", total_ms, None),
        ("prompt_embedding", prompt_embedding_ms, None),
        ("vae_encode", vae_encode_ms, vae_encode_count if vae_encode_count > 0 else None),
        ("denoising_step", denoising_step_ms, denoising_step_count),
        ("decode_latent", decode_latent_ms, None),
    ):
        stats = summarize(values, count_override=count_override)
        print(f"{name:<20} {stats['count']:>8} {stats['mean_ms']:>12.3f} {stats['max_ms']:>12.3f}")
    if results:
        print(f"\nDenoising calls per iteration: {results[0].denoising_calls}")


def save_results(path: str, args: argparse.Namespace, pipe: Any, results: list[IterationMetrics]) -> None:
    payload = {
        "config": {
            "model": args.model,
            "mode": args.mode,
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "strength": args.strength,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "seed": args.seed,
            "pipeline_dtype": str(getattr(pipe, "_benchmark_pipeline_dtype", None)),
            "compile_enabled": not args.disable_compile,
            "compile_mode": args.compile_mode,
            "fullgraph": False,
        },
        "aggregate": {
            "total_ms": summarize([result.total_ms for result in results]),
            "prompt_embedding_ms": summarize([result.prompt_embedding_ms for result in results]),
            "vae_encode_ms": summarize(
                [result.vae_encode_ms for result in results if result.vae_encode_calls > 0],
                count_override=sum(result.vae_encode_calls for result in results) or None,
            ),
            "denoising_step_ms": summarize(
                [result.denoising_step_ms for result in results],
                count_override=sum(result.denoising_calls for result in results),
            ),
            "decode_latent_ms": summarize([result.decode_latent_ms for result in results]),
        },
        "iterations": [asdict(result) for result in results],
    }
    output_path = ensure_parent_dir(path)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0")

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark assumes a single CUDA device, but CUDA is not available.")
    torch.cuda.set_device(0)

    pipe = load_pipeline(
        model_id=args.model,
        mode=args.mode,
        device=device,
    )
    apply_compile(
        pipe=pipe,
        enabled=not args.disable_compile,
        compile_mode=args.compile_mode,
    )

    init_image = load_input_image(args.image, args.width, args.height) if args.mode == "i2i" else None
    call_kwargs = build_call_kwargs(pipe=pipe, args=args, init_image=init_image)
    stage_timer = StageTimer(device=device)

    with instrument_pipeline(pipe=pipe, stage_timer=stage_timer):
        for _ in range(args.warmup):
            _, _ = run_once(
                pipe=pipe,
                base_call_kwargs=call_kwargs,
                stage_timer=stage_timer,
                device=device,
                seed=args.seed,
            )

        measured_runs = [
            run_once(
                pipe=pipe,
                base_call_kwargs=call_kwargs,
                stage_timer=stage_timer,
                device=device,
                seed=args.seed,
            )
            for _ in range(args.iterations)
        ]

    results = [metrics for metrics, _ in measured_runs]
    print_summary(results)

    if args.save_json:
        save_results(args.save_json, args, pipe, results)
        print(f"Saved benchmark json to: {args.save_json}")

    if args.output:
        save_output_image(measured_runs[-1][1], args.output)
        print(f"Saved last iteration image to: {args.output}")


if __name__ == "__main__":
    main()
