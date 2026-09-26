from __future__ import annotations

import contextlib
import hashlib
from typing import Dict, Mapping, Sequence

import torch

from .common import AnalysisContext

DenseUpdates = Dict[str, torch.Tensor]


def _axis_stack(adapter, active: Sequence[int]) -> torch.Tensor:
    return torch.stack([
        adapter.scale * adapter.axis_B[axis].float().cpu() @ adapter.axis_A[axis].float().cpu()
        for axis in active
    ])


def _zero_updates(ctx: AnalysisContext) -> DenseUpdates:
    return {key: torch.zeros(adapter.d_out, adapter.d_in, dtype=ctx.dtype)
            for key, adapter in ctx.adapters}


def task_arithmetic(ctx: AnalysisContext, active: Sequence[int]) -> DenseUpdates:
    if not active:
        return _zero_updates(ctx)
    return {key: _axis_stack(adapter, active).sum(0).to(ctx.dtype)
            for key, adapter in ctx.adapters}


def _trim(stack: torch.Tensor, density: float) -> torch.Tensor:
    if not 0.0 < density <= 1.0:
        raise ValueError("TIES density must lie in (0, 1].")
    flat = stack.flatten(1)
    keep = max(1, round(flat.shape[1] * density))
    indices = flat.abs().topk(keep, dim=1, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return (flat * mask).reshape_as(stack)


def ties_merge_tensor(stack: torch.Tensor, density: float = 0.2, scale: float = 1.0) -> torch.Tensor:
    """TIES trim, mass-sign election, and disjoint mean for one parameter tensor."""
    if stack.shape[0] == 1:
        return scale * stack[0]
    trimmed = _trim(stack, density)
    elected = trimmed.sum(0).sign()
    aligned = (trimmed.sign() == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)
    count = aligned.sum(0).clamp_min(1)
    merged = (trimmed * aligned).sum(0) / count
    return scale * merged


def ties(ctx: AnalysisContext, active: Sequence[int],
         density: float = 0.2, scale: float = 1.0) -> DenseUpdates:
    if not active:
        return _zero_updates(ctx)
    return {key: ties_merge_tensor(_axis_stack(adapter, active), density, scale).to(ctx.dtype)
            for key, adapter in ctx.adapters}


def dare_stack(stack: torch.Tensor, drop_rate: float, generator: torch.Generator) -> torch.Tensor:
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError("DARE drop rate must lie in [0, 1).")
    if drop_rate == 0:
        return stack
    keep = torch.rand(stack.shape, generator=generator, device=stack.device) >= drop_rate
    return stack * keep / (1.0 - drop_rate)


def _dare_axis(delta: torch.Tensor, drop_rate: float, seed: int, key: str, axis: int) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}:{key}:{axis}".encode()).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2 ** 63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    return dare_stack(delta.unsqueeze(0), drop_rate, generator)[0]


def dare_task_arithmetic(ctx: AnalysisContext, active: Sequence[int],
                         drop_rate: float, seed: int) -> DenseUpdates:
    if not active:
        return _zero_updates(ctx)
    result = {}
    for key, adapter in ctx.adapters:
        dense = _axis_stack(adapter, active)
        sparse = torch.stack([_dare_axis(delta, drop_rate, seed, key, axis)
                              for delta, axis in zip(dense, active)])
        result[key] = sparse.sum(0).to(ctx.dtype)
    return result


def dare_ties(ctx: AnalysisContext, active: Sequence[int], drop_rate: float,
              density: float, scale: float, seed: int) -> DenseUpdates:
    if not active:
        return _zero_updates(ctx)
    result = {}
    for key, adapter in ctx.adapters:
        dense = _axis_stack(adapter, active)
        sparse = torch.stack([_dare_axis(delta, drop_rate, seed, key, axis)
                              for delta, axis in zip(dense, active)])
        result[key] = ties_merge_tensor(sparse, density, scale).to(ctx.dtype)
    return result


def knots_basis(adapter, active: Sequence[int]):
    """Compute KnOTS' shared SVD without materializing [Delta_1; ...; Delta_n]."""
    A_parts = [adapter.axis_A[axis].float().cpu() for axis in active]
    B_parts = [adapter.scale * adapter.axis_B[axis].float().cpu() for axis in active]
    B_cat = torch.cat(B_parts, dim=1)
    A_block = torch.block_diag(*A_parts)
    q_b, r_b = torch.linalg.qr(B_cat, mode="reduced")
    q_a, r_a = torch.linalg.qr(A_block.T, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)
    u = q_b @ u_core
    vh = vh_core @ q_a.T
    width = A_parts[0].shape[1]
    v_tasks = torch.stack([vh[:, index * width:(index + 1) * width]
                           for index in range(len(active))])
    return u, singular, v_tasks


def knots_ties(ctx: AnalysisContext, active: Sequence[int], density: float = 0.2,
               scale: float = 1.0) -> DenseUpdates:
    if not active:
        return _zero_updates(ctx)
    result = {}
    for key, adapter in ctx.adapters:
        u, singular, v_tasks = knots_basis(adapter, active)
        merged_v = ties_merge_tensor(v_tasks, density, scale)
        result[key] = ((u * singular.unsqueeze(0)) @ merged_v).to(ctx.dtype)
    return result


def layerwise_distance(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict:
    by_layer = {}
    global_sq = 0.0
    for key in left:
        value = float((left[key].float() - right[key].float()).square().sum())
        layer = key.split(".", 1)[0].removeprefix("layer")
        by_layer[layer] = by_layer.get(layer, 0.0) + value
        global_sq += value
    return {"global_frobenius": global_sq ** 0.5,
            "by_layer_frobenius": {layer: value ** 0.5 for layer, value in sorted(by_layer.items(), key=lambda x: int(x[0]))}}


def distance_to_learned(ctx: AnalysisContext, combination: Sequence[int], merged: Mapping[str, torch.Tensor]) -> dict:
    s = torch.tensor(combination, device=ctx.device, dtype=ctx.dtype).reshape(1, -1)
    s0 = torch.zeros_like(s)
    by_layer = {}
    global_sq = 0.0
    with torch.no_grad():
        for key, adapter in ctx.adapters:
            learned = adapter.scale * (adapter.compute_delta_W(s)[0] - adapter.compute_delta_W(s0)[0])
            error_sq = float((learned.float().cpu() - merged[key].float()).square().sum())
            layer = key.split(".", 1)[0].removeprefix("layer")
            by_layer[layer] = by_layer.get(layer, 0.0) + error_sq
            global_sq += error_sq
    return {"global_frobenius": global_sq ** 0.5,
            "by_layer_frobenius": {layer: value ** 0.5
                                     for layer, value in sorted(by_layer.items(), key=lambda x: int(x[0]))}}


class DenseDeltaHooks:
    """Add a dense endpoint-merge delta after each adapted linear's neutral output."""
    def __init__(self, model, updates: Mapping[str, torch.Tensor]):
        self.handles = []
        for key, delta in updates.items():
            module = model._adapted_linears[key]
            self.handles.append(module.register_forward_hook(self._hook(delta)))

    @staticmethod
    def _hook(delta: torch.Tensor):
        def add(_module, inputs, output):
            x = inputs[0]
            prepared = delta.to(device=x.device, dtype=x.dtype)
            if prepared.ndim == 2:
                correction = torch.nn.functional.linear(x, prepared)
            elif prepared.ndim == 3 and x.ndim == 3 and prepared.shape[0] == x.shape[0]:
                correction = torch.einsum("bti,boi->bto", x, prepared)
            elif prepared.ndim == 3 and x.ndim == 2 and prepared.shape[0] == x.shape[0]:
                correction = torch.einsum("bi,boi->bo", x, prepared)
            else:
                raise ValueError(
                    f"Dense update {tuple(prepared.shape)} cannot apply to {tuple(x.shape)}"
                )
            return output + correction
        return add

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
