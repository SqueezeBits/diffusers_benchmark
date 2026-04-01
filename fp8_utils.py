from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open

_FP8_MAX = 448.0
_WSCALE_SUFFIXES = ["weight_scale", "w_scale", "scale"]
_ISCALE_SUFFIXES = ["input_scale", "act_scale", "x_scale"]


class FP8Linear(nn.Module):
    """Drop-in nn.Linear replacement backed by torch._scaled_mm."""

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        input_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if weight_fp8.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"Expected float8_e4m3fn weights, got {weight_fp8.dtype}"
            )

        self.out_features = weight_fp8.shape[0]
        self.in_features = weight_fp8.shape[1]

        self.register_buffer("weight", weight_fp8)
        self.register_buffer("weight_scale", weight_scale.float().view(1))
        self.register_buffer("input_scale", input_scale.float().view(1))
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        x_fp8 = (x_2d.float() / self.input_scale).clamp(
            -_FP8_MAX, _FP8_MAX
        ).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        return out.reshape(*original_shape[:-1], self.out_features)


def _load_safetensors_dir(cache_dir: str | Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for st_path in sorted(Path(cache_dir).rglob("*.safetensors")):
        with safe_open(str(st_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensors[key] = handle.get_tensor(key)
    return tensors


def download_fp8_weights(model_id: str) -> dict[str, torch.Tensor]:
    print(f"Loading fp8 DiT weights from {model_id}...")
    cache_dir = snapshot_download(model_id)
    tensors = _load_safetensors_dir(cache_dir)
    print(f"Loaded {len(tensors)} fp8 tensors")
    return tensors


def _set_nested_attr(root: nn.Module, dotted_name: str, value: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def _scalar_tensor(
    fp8_tensors: dict[str, torch.Tensor], key: str
) -> torch.Tensor | None:
    tensor = fp8_tensors.get(key)
    if tensor is None:
        return None
    return tensor.float().view(1)


def _find_weight_scale(
    fp8_tensors: dict[str, torch.Tensor], base_key: str
) -> torch.Tensor:
    for suffix in _WSCALE_SUFFIXES:
        scale = _scalar_tensor(fp8_tensors, f"{base_key}.{suffix}")
        if scale is not None:
            return scale
    return torch.ones(1, dtype=torch.float32)


def _find_input_scale(
    fp8_tensors: dict[str, torch.Tensor], base_key: str
) -> torch.Tensor:
    for suffix in _ISCALE_SUFFIXES:
        scale = _scalar_tensor(fp8_tensors, f"{base_key}.{suffix}")
        if scale is not None:
            return scale
    return torch.ones(1, dtype=torch.float32)


def replace_dit_linear_with_fp8(
    transformer: nn.Module,
    fp8_tensors: dict[str, torch.Tensor],
) -> tuple[nn.Module, int]:
    def normalize_key(key: str) -> str:
        return re.sub(r"^transformer\.", "", key)

    normalized_index = {normalize_key(key): key for key in fp8_tensors}
    replaced = 0
    missing_input_scale = 0

    for module_name, module in list(transformer.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        candidates = [
            f"{module_name}.weight",
            f"transformer.{module_name}.weight",
        ]
        fp8_weight_key: str | None = None
        for candidate in candidates:
            if candidate in fp8_tensors:
                fp8_weight_key = candidate
                break
            normalized = normalize_key(candidate)
            if normalized in normalized_index:
                fp8_weight_key = normalized_index[normalized]
                break

        if fp8_weight_key is None:
            continue

        weight = fp8_tensors[fp8_weight_key]
        if weight.dtype != torch.float8_e4m3fn:
            continue

        base_key = fp8_weight_key[: -len(".weight")]
        input_scale = _find_input_scale(fp8_tensors, base_key)
        weight_scale = _find_weight_scale(fp8_tensors, base_key)
        if input_scale.item() == 1.0 and not any(
            f"{base_key}.{suffix}" in fp8_tensors for suffix in _ISCALE_SUFFIXES
        ):
            missing_input_scale += 1

        bias = (
            module.bias.detach().clone().to(torch.bfloat16)
            if module.bias is not None
            else None
        )
        fp8_linear = FP8Linear(
            weight_fp8=weight,
            weight_scale=weight_scale,
            input_scale=input_scale,
            bias=bias,
        )
        _set_nested_attr(transformer, module_name, fp8_linear)
        replaced += 1

    if missing_input_scale > 0:
        print(
            "[fp8] WARNING: "
            f"{missing_input_scale} layers were missing input_scale and defaulted to 1.0"
        )

    return transformer, replaced
