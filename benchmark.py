#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import io
import json
import time
import urllib.request
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


DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_NUM_FRAMES = 81
DEFAULT_GUIDANCE_SCALE = 4.0
DEFAULT_PROMPT = "A cat in a garden"
DEFAULT_WAN_PROMPT = "A cat surfing on a wave"
DEFAULT_NEGATIVE_PROMPT = "low quality"
I2V_IMAGE_URL = (
    "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B"
    "/resolve/main/examples/i2v_input.JPG"
)

WAN_MODEL_PRESETS = {
    "wan2.2-t2v-a14b": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "cls": "WanPipeline",
        "mode": "t2v",
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright "
            "gloves fight intensely on a spotlighted stage."
        ),
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "height": 1280,
        "width": 720,
        "num_frames": 81,
    },
    "wan2.2-i2v-a14b": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "cls": "WanImageToVideoPipeline",
        "mode": "i2v",
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "prompt": DEFAULT_WAN_PROMPT,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "height": 720,
        "width": 1280,
        "num_frames": 81,
    },
    "wan2.1-t2v-14b": {
        "repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "cls": "WanPipeline",
        "mode": "t2v",
        "guidance_scale": 5.0,
        "guidance_scale_2": None,
        "prompt": (
            "Two anthropomorphic cats in comfy boxing gear and bright "
            "gloves fight intensely on a spotlighted stage."
        ),
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "height": 720,
        "width": 1280,
        "num_frames": 81,
    },
    "wan2.1-i2v-14b": {
        "repo": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "cls": "WanImageToVideoPipeline",
        "mode": "i2v",
        "guidance_scale": 5.0,
        "guidance_scale_2": None,
        "prompt": DEFAULT_WAN_PROMPT,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "height": 720,
        "width": 1280,
        "num_frames": 81,
    },
}


@dataclass
class IterationMetrics:
    total_ms: float
    components: dict[str, tuple[int, float]]


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

    def summary(self) -> dict[str, tuple[int, float]]:
        return {name: (len(values), sum(values)) for name, values in self.samples.items()}


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
        description="Benchmark FLUX and WAN diffusers pipelines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id or supported WAN preset name.")
    parser.add_argument("--mode", choices=("t2i", "i2i", "t2v", "i2v"), help="Benchmark mode.")
    parser.add_argument("--prompt", help="Prompt text.")
    parser.add_argument("--negative-prompt", help="Negative prompt text.")
    parser.add_argument("--image", help="Init image path or URL for i2i/i2v mode.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Output height.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Output width.")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Output frame count for video pipelines.")
    parser.add_argument("--strength", type=float, default=0.6, help="Img2Img strength when supported by the pipeline.")
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE, help="Guidance scale.")
    parser.add_argument("--guidance-scale-2", type=float, help="Second guidance scale for supported WAN models.")
    parser.add_argument("--num-inference-steps", type=int, default=28, help="Number of denoising steps.")
    parser.add_argument("--iterations", type=int, default=5, help="Measured iterations.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations run before measurement.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed.")
    parser.add_argument("--output", help="Optional path to save the last measured image or video.")
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile. By default text encoder, DiT, and VAE are compiled.",
    )
    parser.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        help="torch.compile mode passed to compiled components.",
    )
    parser.add_argument(
        "--save-json",
        help="Optional path to save aggregate and per-iteration benchmark results.",
    )
    return parser.parse_args()


def infer_model_family(model_id: str) -> str:
    if model_id in WAN_MODEL_PRESETS:
        return "wan"
    model_id_lower = model_id.lower()
    if "wan" in model_id_lower:
        return "wan"
    if "flux.2-klein" in model_id_lower:
        return "flux2_klein"
    if "flux.2" in model_id_lower:
        return "flux2"
    if "flux.1" in model_id_lower:
        return "flux1"
    raise ValueError(f"Unsupported model family for: {model_id}")


def patch_umt5() -> None:
    from transformers import UMT5EncoderModel

    if getattr(UMT5EncoderModel, "_benchmark_patch_applied", False):
        return

    original_init = UMT5EncoderModel.__init__

    def patched(self, config):
        original_init(self, config)
        self.encoder.embed_tokens = self.shared

    UMT5EncoderModel.__init__ = patched
    UMT5EncoderModel._benchmark_patch_applied = True


def resolve_run_config(args: argparse.Namespace) -> dict[str, Any]:
    family = infer_model_family(args.model)
    config: dict[str, Any] = {
        "model_name": args.model,
        "model_id": args.model,
        "family": family,
        "pipeline_class_name": None,
        "mode": args.mode,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "image": args.image,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "num_inference_steps": args.num_inference_steps,
    }

    if args.model in WAN_MODEL_PRESETS:
        preset = WAN_MODEL_PRESETS[args.model]
        config["model_id"] = preset["repo"]
        config["pipeline_class_name"] = preset["cls"]
        config["mode"] = config["mode"] or preset["mode"]
        config["prompt"] = config["prompt"] or preset["prompt"]
        config["negative_prompt"] = config["negative_prompt"] or preset["negative_prompt"]
        if args.height == DEFAULT_HEIGHT:
            config["height"] = preset["height"]
        if args.width == DEFAULT_WIDTH:
            config["width"] = preset["width"]
        if args.num_frames == DEFAULT_NUM_FRAMES:
            config["num_frames"] = preset["num_frames"]
        if args.guidance_scale == DEFAULT_GUIDANCE_SCALE:
            config["guidance_scale"] = preset["guidance_scale"]
        if config["guidance_scale_2"] is None:
            config["guidance_scale_2"] = preset["guidance_scale_2"]

    if family == "wan":
        config["mode"] = config["mode"] or "t2v"
        config["prompt"] = config["prompt"] or DEFAULT_WAN_PROMPT
        config["negative_prompt"] = config["negative_prompt"] or DEFAULT_NEGATIVE_PROMPT
        if config["mode"] not in {"t2v", "i2v"}:
            raise ValueError("WAN models only support --mode t2v or --mode i2v.")
        if config["mode"] == "i2v" and not config["image"]:
            config["image"] = I2V_IMAGE_URL
    else:
        config["mode"] = config["mode"] or "t2i"
        config["prompt"] = config["prompt"] or DEFAULT_PROMPT
        if config["mode"] not in {"t2i", "i2i"}:
            raise ValueError("FLUX models only support --mode t2i or --mode i2i.")
        if config["mode"] == "i2i" and not config["image"]:
            raise ValueError("--image is required when --mode i2i is selected.")

    return config


def load_pipeline(model_config: dict[str, Any], device: torch.device):
    family = model_config["family"]
    mode = model_config["mode"]

    if family == "wan":
        import diffusers

        patch_umt5()
        class_name = model_config["pipeline_class_name"] or (
            "WanImageToVideoPipeline" if mode == "i2v" else "WanPipeline"
        )
        pipeline_cls = getattr(diffusers, class_name)
    else:
        pipeline_cls = {
            "flux1": FluxPipeline if mode == "t2i" else FluxImg2ImgPipeline,
            "flux2": Flux2Pipeline,
            "flux2_klein": Flux2KleinPipeline,
        }[family]

    pipe = pipeline_cls.from_pretrained(model_config["model_id"], torch_dtype=torch.bfloat16)
    pipe.to(device=device, dtype=torch.bfloat16)
    pipe._benchmark_pipeline_dtype = torch.bfloat16
    pipe._benchmark_family = family
    return pipe


def load_input_image(path: str, width: int, height: int) -> Image.Image:
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path) as response:
            image = Image.open(io.BytesIO(response.read())).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    return image.resize((width, height))


def apply_compile(
    pipe: Any,
    enabled: bool,
    compile_mode: str,
) -> None:
    if not enabled:
        return

    family = getattr(pipe, "_benchmark_family", "flux")

    def compile_callable(fn: Any, *, mode: str = compile_mode) -> Any:
        return torch.compile(fn, mode=mode, dynamic=True, fullgraph=False)

    def compile_module(module: Any, *, mode: str = compile_mode) -> Any:
        return torch.compile(module, mode=mode, dynamic=True, fullgraph=False)

    if family == "wan":
        if getattr(pipe, "text_encoder", None) is not None:
            print("Compiling text_encoder...")
            pipe.text_encoder = compile_module(pipe.text_encoder)
        if getattr(pipe, "transformer", None) is not None:
            compile_target_mode = "max-autotune-no-cudagraphs"
            print("Compiling transformer...")
            pipe.transformer = compile_module(pipe.transformer, mode=compile_target_mode)
        if getattr(pipe, "transformer_2", None) is not None:
            print("Compiling transformer_2...")
            pipe.transformer_2 = compile_module(pipe.transformer_2, mode="max-autotune-no-cudagraphs")
        if getattr(pipe, "vae", None) is not None:
            if hasattr(pipe.vae, "encode"):
                print("Compiling vae.encode...")
                pipe.vae.encode = compile_callable(pipe.vae.encode)
            if hasattr(pipe.vae, "decode"):
                print("Compiling vae.decode...")
                pipe.vae.decode = compile_callable(pipe.vae.decode)
        return

    for attr_name in ("text_encoder", "text_encoder_2"):
        module = getattr(pipe, attr_name, None)
        if module is not None:
            print(f"Compiling {attr_name}...")
            module.forward = compile_callable(module.forward)

    if getattr(pipe, "vae", None) is not None and family != "wan":
        if hasattr(pipe.vae, "encode"):
            print("Compiling vae.encode...")
            pipe.vae.encode = compile_callable(pipe.vae.encode)
        if hasattr(pipe.vae, "decode"):
            print("Compiling vae.decode...")
            pipe.vae.decode = compile_callable(pipe.vae.decode)


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

    family = getattr(pipe, "_benchmark_family", "flux")
    if family == "wan":
        patches.append(MethodPatch(pipe, "encode_prompt", wrap_stage("text_encoder", pipe.encode_prompt)))
        patches.append(MethodPatch(pipe.transformer, "forward", wrap_stage("transformer", pipe.transformer.forward)))
        if getattr(pipe, "transformer_2", None) is not None:
            patches.append(
                MethodPatch(pipe.transformer_2, "forward", wrap_stage("transformer_2", pipe.transformer_2.forward))
            )
        if hasattr(pipe.vae, "encode"):
            patches.append(MethodPatch(pipe.vae, "encode", wrap_stage("vae_encode", pipe.vae.encode)))
        if hasattr(pipe.vae, "decode"):
            patches.append(MethodPatch(pipe.vae, "decode", wrap_stage("vae_decode", pipe.vae.decode)))
        if hasattr(pipe, "prepare_latents"):
            patches.append(MethodPatch(pipe, "prepare_latents", wrap_stage("prepare_latents", pipe.prepare_latents)))
    else:
        patches.append(MethodPatch(pipe, "encode_prompt", wrap_stage("prompt_embedding", pipe.encode_prompt)))
        patches.append(MethodPatch(pipe.transformer, "forward", wrap_stage("denoising_step", pipe.transformer.forward)))
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
            patches.append(MethodPatch(pipe.image_processor, "postprocess", wrap_decode_end(pipe.image_processor.postprocess)))

    try:
        yield
    finally:
        for patch in reversed(patches):
            patch.restore()


def build_call_kwargs(
    pipe: Any,
    model_config: dict[str, Any],
    init_image: Image.Image | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": model_config["prompt"],
        "height": model_config["height"],
        "width": model_config["width"],
        "num_inference_steps": model_config["num_inference_steps"],
        "guidance_scale": model_config["guidance_scale"],
    }
    signature = inspect.signature(pipe.__call__)

    if model_config["family"] == "wan":
        kwargs["negative_prompt"] = model_config["negative_prompt"]
        kwargs["num_frames"] = model_config["num_frames"]
        kwargs["output_type"] = "np"
        if model_config["guidance_scale_2"] is not None and "guidance_scale_2" in signature.parameters:
            kwargs["guidance_scale_2"] = model_config["guidance_scale_2"]
        if model_config["mode"] == "i2v":
            kwargs["image"] = init_image
    else:
        kwargs["return_dict"] = False
        if model_config["mode"] == "i2i":
            kwargs["image"] = init_image
            if "strength" in signature.parameters:
                kwargs["strength"] = model_config["strength"]

    return kwargs


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def collect_iteration_metrics(stage_timer: StageTimer, total_ms: float) -> IterationMetrics:
    return IterationMetrics(total_ms=total_ms, components=stage_timer.summary())


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


def extract_first_frames(output: Any) -> np.ndarray:
    frames = getattr(output, "frames", output)
    if isinstance(frames, tuple):
        frames = frames[0]
    if isinstance(frames, list):
        if not frames:
            raise ValueError("Pipeline returned an empty frame list.")
        frames = frames[0]
    if isinstance(frames, np.ndarray):
        while frames.ndim > 4 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.ndim != 4:
            raise ValueError(f"Expected video frames with ndim 4 but got shape {frames.shape}.")
        return frames
    raise TypeError(f"Unsupported video output type: {type(frames)}")


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


def save_output_video(output: Any, path: str, width: int, height: int) -> None:
    import av

    output_path = ensure_parent_dir(path)
    frames = extract_first_frames(output)
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("h264", rate=16)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_np in frames:
            frame_u8 = (np.clip(frame_np, 0, 1) * 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(frame_u8, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


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


def build_aggregate_components(results: list[IterationMetrics]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    stage_names = sorted({name for result in results for name in result.components})
    for stage_name in stage_names:
        total_calls = sum(result.components.get(stage_name, (0, 0.0))[0] for result in results)
        total_times = [result.components.get(stage_name, (0, 0.0))[1] for result in results]
        total_time = sum(total_times)
        payload[stage_name] = {
            "count": total_calls,
            "mean_total_ms": mean(total_times) if total_times else 0.0,
            "max_total_ms": max(total_times) if total_times else 0.0,
            "mean_call_ms": total_time / total_calls if total_calls else 0.0,
        }
    return payload


def print_summary(results: list[IterationMetrics]) -> None:
    print("\nBenchmark results (ms)")
    print(f"{'stage':<20} {'count':>8} {'avg/call':>12} {'avg total':>12}")
    print(f"{'-' * 20} {'-' * 8} {'-' * 12} {'-' * 12}")
    total_stats = summarize([result.total_ms for result in results])
    print(f"{'total':<20} {total_stats['count']:>8} {total_stats['mean_ms']:>12.3f} {total_stats['mean_ms']:>12.3f}")

    stage_names = sorted({name for result in results for name in result.components})
    for stage_name in stage_names:
        total_calls = sum(result.components.get(stage_name, (0, 0.0))[0] for result in results)
        total_time = sum(result.components.get(stage_name, (0, 0.0))[1] for result in results)
        per_iteration_totals = [result.components.get(stage_name, (0, 0.0))[1] for result in results]
        avg_call = total_time / total_calls if total_calls else 0.0
        avg_total = mean(per_iteration_totals) if per_iteration_totals else 0.0
        print(f"{stage_name:<20} {total_calls:>8} {avg_call:>12.3f} {avg_total:>12.3f}")


def save_results(
    path: str,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    pipe: Any,
    results: list[IterationMetrics],
) -> None:
    payload = {
        "config": {
            "model": args.model,
            "model_id": model_config["model_id"],
            "family": model_config["family"],
            "mode": model_config["mode"],
            "prompt": model_config["prompt"],
            "negative_prompt": model_config["negative_prompt"],
            "height": model_config["height"],
            "width": model_config["width"],
            "num_frames": model_config["num_frames"],
            "strength": model_config["strength"],
            "guidance_scale": model_config["guidance_scale"],
            "guidance_scale_2": model_config["guidance_scale_2"],
            "num_inference_steps": model_config["num_inference_steps"],
            "iterations": args.iterations,
            "warmup": args.warmup,
            "seed": args.seed,
            "pipeline_dtype": str(getattr(pipe, "_benchmark_pipeline_dtype", None)),
            "compile_enabled": not args.disable_compile,
            "compile_mode": args.compile_mode,
        },
        "aggregate": {
            "total_ms": summarize([result.total_ms for result in results]),
            "components": build_aggregate_components(results),
        },
        "iterations": [asdict(result) for result in results],
    }
    output_path = ensure_parent_dir(path)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    model_config = resolve_run_config(args)
    device = torch.device("cuda:0")

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark assumes a single CUDA device, but CUDA is not available.")
    torch.cuda.set_device(0)

    pipe = load_pipeline(model_config=model_config, device=device)
    apply_compile(
        pipe=pipe,
        enabled=not args.disable_compile,
        compile_mode=args.compile_mode,
    )

    init_image = None
    if model_config["mode"] in {"i2i", "i2v"}:
        init_image = load_input_image(model_config["image"], model_config["width"], model_config["height"])
    call_kwargs = build_call_kwargs(pipe=pipe, model_config=model_config, init_image=init_image)
    stage_timer = StageTimer(device=device)

    with instrument_pipeline(pipe=pipe, stage_timer=stage_timer):
        for _ in range(args.warmup):
            warmup_call_kwargs = dict(call_kwargs)
            warmup_call_kwargs["num_inference_steps"] = 2
            if model_config["family"] == "wan":
                warmup_call_kwargs["num_frames"] = 17
            run_once(pipe=pipe, base_call_kwargs=warmup_call_kwargs, stage_timer=stage_timer, device=device, seed=args.seed)

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
        save_results(args.save_json, args, model_config, pipe, results)
        print(f"Saved benchmark json to: {args.save_json}")

    if args.output:
        if model_config["family"] == "wan":
            save_output_video(measured_runs[-1][1], args.output, model_config["width"], model_config["height"])
            print(f"Saved last iteration video to: {args.output}")
        else:
            save_output_image(measured_runs[-1][1], args.output)
            print(f"Saved last iteration image to: {args.output}")


if __name__ == "__main__":
    main()
