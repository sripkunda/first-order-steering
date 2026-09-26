from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .merging import dare_stack, ties_merge_tensor


_LANGUAGE_KEY = re.compile(
    r"\.(?:language_model|model)\.layers\.(\d+)\.(?:self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_([AB])(?:\.default)?\.weight$"
)


@dataclass(frozen=True)
class IndependentFactor:
    A: torch.Tensor
    B: torch.Tensor
    scale: float

    def dense(self) -> torch.Tensor:
        return self.scale * (self.B.float() @ self.A.float())


@dataclass
class IndependentLoRAs:
    root: Path
    axes: list[str]
    base_model: str
    factors: Dict[str, Dict[str, IndependentFactor]]
    _device_cache: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def keys(self) -> list[str]:
        return sorted(set().union(*(set(current) for current in self.factors.values())))

    def reference_factor(self, key: str) -> IndependentFactor:
        for axis in self.axes:
            factor = self.factors[axis].get(key)
            if factor is not None:
                return factor
        raise KeyError(f"No independent adapter contains projection {key!r}")

    def validate_uniform_subset(self, expected_keys: Sequence[str]) -> list[str]:
        """Require every axis to use the same nonempty subset of model projections."""
        by_axis = {axis: set(self.factors[axis]) for axis in self.axes}
        reference = by_axis[self.axes[0]]
        mismatched = {
            axis: sorted(keys) for axis, keys in by_axis.items() if keys != reference
        }
        if mismatched:
            raise ValueError(
                "Independent axes do not share one attachment set: "
                f"{mismatched}"
            )
        unexpected = sorted(reference - set(expected_keys))
        if unexpected:
            raise ValueError(
                f"Independent adapters contain unknown model projections: {unexpected}"
            )
        if not reference:
            raise ValueError("Independent adapter attachment set is empty")
        return sorted(reference)

    def dense_axis(self, axis: int | str) -> Dict[str, torch.Tensor]:
        name = self.axes[axis] if isinstance(axis, int) else axis
        return {key: factor.dense() for key, factor in self.factors[name].items()}

    def prepared_factor(self, axis: str, key: str, device: torch.device,
                        dtype: torch.dtype) -> IndependentFactor:
        cache_key = (axis, key, str(device), dtype)
        if cache_key not in self._device_cache:
            factor = self.factors[axis][key]
            # Teacher caching calls this method under inference_mode. Tensors
            # created there cannot later be saved for backward, even when they
            # are fixed constants. The interaction fit must backpropagate
            # through task-arithmetic projections into earlier R_S branches,
            # so explicitly construct ordinary detached cache tensors.
            with torch.inference_mode(False):
                A = factor.A.to(
                    device=device, dtype=dtype, non_blocking=True,
                ).detach().clone()
                B = factor.B.to(
                    device=device, dtype=dtype, non_blocking=True,
                ).detach().clone()
            self._device_cache[cache_key] = IndependentFactor(A, B, factor.scale)
        return self._device_cache[cache_key]

    def stack(self, key: str, active: Sequence[int]) -> torch.Tensor:
        reference = self.reference_factor(key).dense()
        return torch.stack([
            self.factors[self.axes[axis]][key].dense()
            if key in self.factors[self.axes[axis]] else torch.zeros_like(reference)
            for axis in active
        ])

    def task_arithmetic(self, values: Sequence[float]) -> Dict[str, torch.Tensor]:
        if len(values) != len(self.axes):
            raise ValueError(f"Expected {len(self.axes)} baseline coordinates, got {len(values)}")
        return {
            key: sum(
                float(value) * self.factors[axis][key].dense()
                for axis, value in zip(self.axes, values)
                if key in self.factors[axis]
            )
            for key in self.keys
        }

    def merge(self, method: str, active: Sequence[int], density: float = 0.2,
              scale: float = 1.0, drop_rate: float = 0.9,
              seed: int = 42) -> Dict[str, torch.Tensor]:
        if not active:
            return {key: torch.zeros_like(self.factors[self.axes[0]][key].dense()) for key in self.keys}
        result = {}
        for key in self.keys:
            stack = self.stack(key, active)
            if method == "task_arithmetic":
                merged = stack.sum(0)
            elif method == "ties":
                merged = ties_merge_tensor(stack, density, scale)
            elif method in {"dare_task_arithmetic", "dare_ties"}:
                generator = torch.Generator(device="cpu").manual_seed(
                    _stable_seed(seed, key)
                )
                sparse = dare_stack(stack, drop_rate, generator)
                merged = (sparse.sum(0) if method == "dare_task_arithmetic"
                          else ties_merge_tensor(sparse, density, scale))
            elif method == "knots_ties":
                merged = _knots_ties(self, key, active, density, scale)
            else:
                raise ValueError(f"Unknown independent merge method: {method}")
            result[key] = merged
        return result


class IndependentLoRAHooks:
    """Apply continuous independent task arithmetic without dense matrices."""
    def __init__(self, model, baseline: IndependentLoRAs, values: torch.Tensor,
                 keys: Sequence[str] | None = None):
        if values.ndim != 2 or values.shape[1] != len(baseline.axes):
            raise ValueError("Independent task-arithmetic values must have shape [batch, axes]")
        self.handles = []
        selected_keys = baseline.keys if keys is None else list(keys)
        unknown = sorted(set(selected_keys) - set(baseline.keys))
        if unknown:
            raise ValueError(f"Unknown independent-LoRA projection keys: {unknown}")
        for key in selected_keys:
            module = model._adapted_linears[key]
            weight = module.linear.weight
            factors = [
                (baseline.prepared_factor(axis, key, weight.device, weight.dtype)
                 if key in baseline.factors[axis] else None)
                for axis in baseline.axes
            ]
            self.handles.append(module.register_forward_hook(
                self._hook(factors, values.to(device=weight.device, dtype=weight.dtype))
            ))

    @staticmethod
    def _hook(factors: Sequence[IndependentFactor], values: torch.Tensor):
        def add(_module, inputs, output):
            x = inputs[0]
            coefficients = values
            correction = torch.zeros_like(output)
            for axis, factor in enumerate(factors):
                if factor is None:
                    continue
                reduced = torch.nn.functional.linear(x, factor.A)
                current = torch.nn.functional.linear(
                    reduced, factor.B,
                )
                shape = (coefficients.shape[0],) + (1,) * (current.ndim - 1)
                correction = correction + factor.scale * current * coefficients[:, axis].reshape(shape)
            return output + correction
        return add

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


def interaction_subsets(axis_count: int) -> list[tuple[int, ...]]:
    """All multi-axis subsets in deterministic degree/lexicographic order."""
    if axis_count < 1:
        raise ValueError("axis_count must be positive")
    return [
        subset
        for degree in range(2, axis_count + 1)
        for subset in itertools.combinations(range(axis_count), degree)
    ]


class MultiAffineInteractions(nn.Module):
    """Static low-rank interaction terms on top of exact task arithmetic.

    For every adapted projection ``p`` this module represents

        sum_{|S| >= 2} prod_{i in S} s_i R_{S,p},
        R_{S,p} = scale * B_{S,p} A_{S,p}.

    The product gate is identically zero on the origin and every pure axis, so
    fitting these parameters can never alter any standalone adapter endpoint.
    """

    FORMAT_VERSION = 1

    def __init__(self, baseline: IndependentLoRAs, rank: int = 8, seed: int = 42):
        super().__init__()
        if rank < 1:
            raise ValueError("interaction rank must be positive")
        self.axes = list(baseline.axes)
        self.base_model = baseline.base_model
        self.rank = int(rank)
        self.subsets = interaction_subsets(len(self.axes))
        self.keys = list(baseline.keys)
        self.factors = nn.ModuleDict()
        # A deployable scalar gain for every interaction subset. Historical
        # artifacts implicitly use one. Constrained fitting can attenuate the
        # learned directions without changing the immutable task adapters or
        # the low-rank interaction bases.
        self.register_buffer("subset_gains", torch.ones(len(self.subsets)))

        # A standard LoRA initialization is deterministic here: A is random
        # and B is zero. Thus the initial model is exactly task arithmetic.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for subset in self.subsets:
            for key in self.keys:
                reference = baseline.factors[self.axes[0]][key]
                d_in = int(reference.A.shape[1])
                d_out = int(reference.B.shape[0])
                branch = nn.Module()
                bound = 1.0 / max(1, d_in) ** 0.5
                branch.A = nn.Parameter(
                    torch.empty(self.rank, d_in).uniform_(-bound, bound, generator=generator)
                )
                branch.B = nn.Parameter(torch.zeros(d_out, self.rank))
                self.factors[self._factor_key(subset, key)] = branch

    @staticmethod
    def _factor_key(subset: Sequence[int], key: str) -> str:
        return "s" + "_".join(map(str, subset)) + "__" + key.replace(".", "__")

    @property
    def scale(self) -> float:
        return 1.0 / self.rank

    def branch(self, subset: Sequence[int], key: str) -> nn.Module:
        return self.factors[self._factor_key(subset, key)]

    def subset_gain(self, subset: Sequence[int]) -> torch.Tensor:
        return self.subset_gains[self.subsets.index(tuple(subset))]

    def save_artifact(self, directory: Path, metadata: Mapping | None = None) -> None:
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for subset in self.subsets:
            for key in self.keys:
                branch = self.branch(subset, key)
                prefix = self._factor_key(subset, key)
                tensors[f"{prefix}.A"] = branch.A.detach().cpu().contiguous()
                tensors[f"{prefix}.B"] = branch.B.detach().cpu().contiguous()
        tensors["subset_gains"] = self.subset_gains.detach().cpu().contiguous()
        save_file(tensors, str(directory / "interaction_adapters.safetensors"))
        payload = {
            "format": "multi_affine_distributional_interactions",
            "format_version": self.FORMAT_VERSION,
            "base_model": self.base_model,
            "axes": self.axes,
            "rank": self.rank,
            "scale": self.scale,
            "subsets": [list(subset) for subset in self.subsets],
            "subset_gains": [float(value) for value in self.subset_gains.detach().cpu()],
            "projection_keys": self.keys,
            **dict(metadata or {}),
        }
        (directory / "interaction_config.json").write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load_artifact(cls, directory: Path, baseline: IndependentLoRAs) -> "MultiAffineInteractions":
        from safetensors.torch import load_file

        config = json.loads((directory / "interaction_config.json").read_text())
        if config.get("format") != "multi_affine_distributional_interactions":
            raise ValueError(f"Unsupported interaction artifact in {directory}")
        if config.get("base_model") != baseline.base_model or config.get("axes") != baseline.axes:
            raise ValueError("Interaction artifact does not match the independent adapters")
        result = cls(baseline, rank=int(config["rank"]), seed=0)
        expected_subsets = [list(item) for item in result.subsets]
        if config.get("subsets") != expected_subsets:
            raise ValueError("Interaction subset layout does not match the independent adapters")
        tensors = load_file(str(directory / "interaction_adapters.safetensors"), device="cpu")
        state = {}
        for subset in result.subsets:
            for key in result.keys:
                prefix = result._factor_key(subset, key)
                state[f"factors.{prefix}.A"] = tensors[f"{prefix}.A"]
                state[f"factors.{prefix}.B"] = tensors[f"{prefix}.B"]
        if "subset_gains" in tensors:
            gains = tensors["subset_gains"].float()
        else:
            gains = torch.tensor(
                config.get("subset_gains", [1.0] * len(result.subsets)),
                dtype=torch.float32,
            )
        if tuple(gains.shape) != (len(result.subsets),):
            raise ValueError("Interaction subset gains do not match the subset layout")
        state["subset_gains"] = gains
        result.load_state_dict(state, strict=True)
        return result


class MultiAffineInteractionHooks:
    """Apply fitted multi-affine interaction LoRAs to a steered model wrapper."""

    def __init__(self, model, interactions: MultiAffineInteractions, values: torch.Tensor,
                 subset_gains: torch.Tensor | None = None,
                 keys: Sequence[str] | None = None):
        if values.ndim != 2 or values.shape[1] != len(interactions.axes):
            raise ValueError("Interaction values must have shape [batch, axes]")
        if subset_gains is None:
            subset_gains = interactions.subset_gains
        subset_gains = torch.as_tensor(
            subset_gains, device=values.device, dtype=values.dtype,
        )
        if subset_gains.ndim == 1:
            subset_gains = subset_gains.unsqueeze(0).expand(values.shape[0], -1)
        if tuple(subset_gains.shape) != (values.shape[0], len(interactions.subsets)):
            raise ValueError(
                "Interaction gains must have shape [subsets] or [batch, subsets]"
            )
        self.handles = []
        selected_keys = interactions.keys if keys is None else list(keys)
        unknown = sorted(set(selected_keys) - set(interactions.keys))
        if unknown:
            raise ValueError(f"Unknown interaction projection keys: {unknown}")
        for key in selected_keys:
            module = model._adapted_linears[key]
            weight = module.linear.weight
            coefficients = values.to(device=weight.device, dtype=weight.dtype)
            gains = subset_gains.to(device=weight.device, dtype=weight.dtype)
            branches = [(index, subset, interactions.branch(subset, key))
                        for index, subset in enumerate(interactions.subsets)]
            self.handles.append(module.register_forward_hook(
                self._hook(branches, coefficients, gains, interactions.scale)
            ))

    @staticmethod
    def _hook(branches: Iterable[tuple[int, tuple[int, ...], nn.Module]],
              values: torch.Tensor, gains: torch.Tensor, scale: float):
        branches = list(branches)

        def add(_module, inputs, output):
            x = inputs[0]
            if x.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Interaction signal batch ({values.shape[0]}) does not match "
                    f"activation batch ({x.shape[0]})."
                )
            correction = torch.zeros_like(output)
            shape = (values.shape[0],) + (1,) * (output.ndim - 1)
            for subset_index, subset, branch in branches:
                # Parameters stay FP32 under Adam; casts preserve autograd and
                # keep the projection arithmetic in the model activation dtype.
                A = branch.A.to(device=x.device, dtype=x.dtype)
                B = branch.B.to(device=x.device, dtype=x.dtype)
                coefficient = values[:, list(subset)].prod(dim=-1).reshape(shape)
                gain = gains[:, subset_index].reshape(shape)
                correction = correction + scale * gain * torch.nn.functional.linear(
                    torch.nn.functional.linear(x, A), B,
                ) * coefficient
            return output + correction

        return add

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


def _stable_seed(seed: int, key: str) -> int:
    import hashlib
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)


def _knots_ties(baseline: IndependentLoRAs, key: str, active: Sequence[int],
                 density: float, scale: float) -> torch.Tensor:
    factors = [baseline.factors[baseline.axes[axis]][key] for axis in active]
    A_parts = [factor.A.float() for factor in factors]
    B_parts = [factor.scale * factor.B.float() for factor in factors]
    B_cat = torch.cat(B_parts, dim=1)
    A_block = torch.block_diag(*A_parts)
    q_b, r_b = torch.linalg.qr(B_cat, mode="reduced")
    q_a, r_a = torch.linalg.qr(A_block.T, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)
    u = q_b @ u_core
    vh = vh_core @ q_a.T
    width = A_parts[0].shape[1]
    tasks = torch.stack([
        vh[:, index * width:(index + 1) * width]
        for index in range(len(active))
    ])
    return (u * singular.unsqueeze(0)) @ ties_merge_tensor(tasks, density, scale)


def _latest_complete_axis_dir(root: Path, axis: str) -> Path:
    axis_root = root / axis
    if (axis_root / "adapter_model.safetensors").is_file():
        return axis_root
    candidates = []
    for path in axis_root.glob("checkpoint-*") if axis_root.is_dir() else []:
        if not (path / "adapter_model.safetensors").is_file():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError(f"No complete independent LoRA for axis {axis!r} under {root}")
    return max(candidates)[1]


def load_independent_loras(root: Path, expected_axes: Sequence[str],
                           expected_keys: Sequence[str] | None = None,
                           expected_base_model: str | None = None,
                           allow_sparse: bool = False) -> IndependentLoRAs:
    root = root.resolve()
    factors: Dict[str, Dict[str, IndependentFactor]] = {}
    base_model = None
    for axis in expected_axes:
        source = _latest_complete_axis_dir(root, axis)
        config = json.loads((source / "adapter_config.json").read_text())
        current_base = config["base_model_name_or_path"]
        if expected_base_model is not None and current_base != expected_base_model:
            raise ValueError(
                f"Independent axis {axis!r} uses base {current_base!r}, expected {expected_base_model!r}"
            )
        if base_model is not None and current_base != base_model:
            raise ValueError("Independent axes do not share the same base model")
        base_model = current_base
        default_rank = int(config["r"])
        default_alpha = float(config["lora_alpha"])
        tensors = load_file(str(source / "adapter_model.safetensors"), device="cpu")
        pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        for tensor_key, tensor in tensors.items():
            match = _LANGUAGE_KEY.search(tensor_key)
            if match is None:
                continue
            layer, projection, side = match.groups()
            pairs.setdefault(f"layer{layer}.{projection}", {})[side] = tensor
        axis_factors = {}
        for key, pair in pairs.items():
            if set(pair) != {"A", "B"}:
                raise ValueError(f"Incomplete PEFT factor pair for {axis}:{key}")
            rank = pair["A"].shape[0]
            rank_pattern: Mapping[str, int] = config.get("rank_pattern") or {}
            alpha_pattern: Mapping[str, float] = config.get("alpha_pattern") or {}
            effective_rank = int(rank_pattern.get(key, rank if rank else default_rank))
            effective_alpha = float(alpha_pattern.get(key, default_alpha))
            axis_factors[key] = IndependentFactor(
                pair["A"].float(), pair["B"].float(), effective_alpha / effective_rank,
            )
        if expected_keys is not None:
            missing = sorted(set(expected_keys) - set(axis_factors))
            unexpected = sorted(set(axis_factors) - set(expected_keys))
            if unexpected or (missing and not allow_sparse):
                raise ValueError(
                    f"Independent axis {axis!r} attachment mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        if not axis_factors:
            raise ValueError(f"Independent axis {axis!r} contains no recognized LoRA factors")
        factors[axis] = axis_factors
    return IndependentLoRAs(root, list(expected_axes), str(base_model), factors)
