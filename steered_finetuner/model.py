"""
model.py — Steerable LoRA adapter and SteeredModel.

Architecture (THEORY.md §Steerable LoRA):

  For every linear weight matrix W in {Q, K, V, O, gate, up, down} at each
  decoder layer, the default adapter produces anchored nonlinear axis paths
  plus a gated neural merger:

      ΔW(s) = ΔW_neutral + Σ_i H_i(s_i) + q(s) R(s)

  Each g_i is a small neural network.  Thus every axis can follow a nonlinear
  path through weight space. q(s) is zero on all pure axes and activates R(s)
  only when multiple axes are present.  The learned neutral update means s=0
  is an adapted endpoint rather than a forced base-model bypass.

  Since ΔW(s) depends only on s (not on any forward-pass activations), all
  weight updates are computed once before the forward pass and reused across
  all token positions.  The effective rank grows linearly with the number of
  axes, rather than using the previous full-matrix hypernetwork outputs.

  The effective linear projection for input x ∈ R^{B×T×d_in} is:
      output = x @ W.T + (1/r) * (x @ A(s).T) @ B(s).T

  which avoids materializing the full ΔW matrix.

  Initialization:
      All B factors are zero-initialized → ΔW = 0 at start.
      A factors and neural gates use standard initialization.

  Hook strategy:
      A single pre-forward hook on the model sets the current signal s and
      precomputes all (A_W(s), B_W(s)) pairs before any layer runs.
      Each adapted linear module's forward is replaced by a wrapper that applies
      the LoRA correction using the precomputed factors.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

import logging
logger = logging.getLogger(__name__)

from config import TrainingConfig


# ── Names of the weight matrices to adapt ────────────────────────────────────
# These are the attribute names on each LlamaDecoderLayer's sub-modules.
# attention: q, k, v, o projections
# mlp: gate, up, down projections
_ATTN_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJ_NAMES  = ("gate_proj", "up_proj", "down_proj")


def _decoder_layers(model) -> nn.ModuleList:
    """Return the text decoder layers for supported causal and multimodal LMs."""
    root = model.model
    if hasattr(root, "layers"):
        return root.layers
    # Gemma 4 conditional-generation models hold the text decoder here.
    if hasattr(root, "language_model") and hasattr(root.language_model, "layers"):
        return root.language_model.layers
    raise TypeError(f"Unsupported model backbone: cannot find decoder layers on {type(model).__name__}")


def _text_config(model):
    return getattr(model.config, "text_config", model.config)


def _load_base_model(
    model_id: str,
    dtype: torch.dtype,
    device: str,
    attn_implementation: Optional[str] = None,
):
    """Load a causal or multimodal LM using its config model type."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # Qwen2.5-VL and Gemma 4 are conditional-generation VLMs rather than
    # AutoModelForCausalLM models.  Their language decoder is still exposed
    # through the same layer/projection layout used by the steering wrapper.
    multimodal_types = {"gemma4", "qwen2_5_vl", "qwen3_vl"}
    if config.model_type in multimodal_types or config.model_type.endswith("_vl"):
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 4 requires Transformers with AutoModelForMultimodalLM (>=5.5)."
            ) from exc
        loader = AutoModelForMultimodalLM
    else:
        loader = AutoModelForCausalLM
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if attn_implementation is not None:
        # Probes which require exact higher derivatives must opt out of fused
        # attention kernels.  Leaving this unset preserves the normal training
        # and inference backend choice.
        kwargs["attn_implementation"] = attn_implementation
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = loader.from_pretrained(model_id, **kwargs)
    return model if device == "auto" else model.to(device)


# ── Single adapter pair (φ^A, φ^B) for one weight matrix ─────────────────────

class WeightAdapter(nn.Module):
    """
    Adapter pair for a single weight matrix W of shape (d_out, d_in).

    Given signal s ∈ [0,1]^k:
        A(s) = φ^A(W_s^A · s)   shape (r, d_in)
        B(s) = φ^B(W_s^B · s)   shape (d_out, r)
        ΔW(s) = B(s) @ A(s)     shape (d_out, d_in)

    The LoRA correction to a linear forward pass is applied as:
        output += (1/r) * (x @ A(s).T) @ B(s).T
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        signal_dim: int,
        rank: int,
        adapter_hidden_dim: int,
    ):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.rank  = rank
        self.scale = 1.0 / rank

        # Signal projection: k → d_proj  (separate for A and B)
        # φ^A: Linear(k→d_proj) → GeLU → Linear(d_proj → r*d_in)
        self.phi_A = nn.Sequential(
            nn.Linear(signal_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, rank * d_in),
        )
        # φ^B: Linear(k→d_proj) → GeLU → Linear(d_proj → d_out*r)
        self.phi_B = nn.Sequential(
            nn.Linear(signal_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(adapter_hidden_dim, d_out * rank),
        )

        # Standard Gaussian init for φ^A (mirrors LoRA A init).
        nn.init.kaiming_uniform_(self.phi_A[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.phi_A[2].weight, a=math.sqrt(5))

        # Zero-initialize φ^B output layer so ΔW = 0 at init (mirrors LoRA B init).
        nn.init.zeros_(self.phi_B[2].weight)
        nn.init.zeros_(self.phi_B[2].bias)

    def compute_AB(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute A(s) and B(s) for a batch of signal vectors.

        Args:
            s: (B, k) signal vectors

        Returns:
            A: (B, r, d_in)
            B: (B, d_out, r)
        """
        B_size = s.shape[0]
        A = self.phi_A(s).view(B_size, self.rank, self.d_in)      # (B, r, d_in)
        B = self.phi_B(s).view(B_size, self.d_out, self.rank)     # (B, d_out, r)
        return A, B

    def compute_delta_W(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute ΔW(s) = B(s) @ A(s) for auxiliary losses.
        Does NOT apply the 1/r scaling (caller handles that if needed).

        Args:
            s: (B, k)

        Returns:
            delta_W: (B, d_out, d_in)
        """
        A, B = self.compute_AB(s)
        return torch.bmm(B, A)  # (B, d_out, d_in)

    def apply_to_output(
        self,
        x: torch.Tensor,              # input to the linear layer
        A: torch.Tensor,              # (B, r, d_in) precomputed
        B_mat: torch.Tensor,          # (B, d_out, r) precomputed
    ) -> Optional[torch.Tensor]:
        """
        Compute the LoRA correction term: (1/r) * (x @ A.T) @ B.T

        Returns the correction tensor (same shape as base_out) or None if the
        input shape is unexpected (caller should skip the correction).

        Handles:
          (B, T, d_in) — standard batched sequence
          (T, d_in)    — unbatched sequence (B=1 implicit, uses A[0]/B[0])
        """
        if x.dim() == 3:
            batch_size = x.size(0)
            # Probes commonly apply one steering signal to many prompts. The
            # adapter factors are then (1, ...), while activations are
            # (B, T, ...); broadcast without materializing copies.
            if A.size(0) == 1 and batch_size != 1:
                A = A.expand(batch_size, -1, -1)
                B_mat = B_mat.expand(batch_size, -1, -1)
            elif A.size(0) != batch_size:
                raise ValueError(
                    f"Adapter signal batch ({A.size(0)}) does not match activation batch ({batch_size})."
                )
            # (B, T, d_in)
            xA  = torch.bmm(x, A.transpose(1, 2))           # (B, T, r)
            out = torch.bmm(xA, B_mat.transpose(1, 2))      # (B, T, d_out)
            return self.scale * out

        if x.dim() == 2:
            # (T, d_in) — unbatched; use first (only) batch element of A/B.
            x3  = x.unsqueeze(0)                                 # (1, T, d_in)
            xA  = torch.bmm(x3, A[:1].transpose(1, 2))          # (1, T, r)
            out = torch.bmm(xA, B_mat[:1].transpose(1, 2))      # (1, T, d_out)
            return (self.scale * out).squeeze(0)                 # (T, d_out)

        # Unexpected shape — skip correction silently.
        return None


class AnchoredNeuralMergerWeightAdapter(nn.Module):
    """Anchored per-axis paths with a gated neural multi-axis merger.

    For one base weight matrix, the effective update is

        ΔW(s) = B_0 A_0 + Σ_i B_i diag(c_i(s_i)) A_i
                + q(s) B_R diag(g_R(s)) A_R
        c_i(t) = t 1_r + t(1-t) g_i(t).

    The normalized pair-activation gate q(s) is exactly zero whenever at most
    one signal coordinate is active.  Consequently the merger cannot alter
    any learned pure-axis path, but can bend the combined trajectory when two
    or more axes are active.  Concatenating factors applies the entire sum as
    one correction without materialising ΔW.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        signal_dim: int,
        rank: int,
        adapter_hidden_dim: int,
    ):
        super().__init__()
        if signal_dim < 1:
            raise ValueError("additive_axis adapters require signal_dim >= 1")
        self.d_in = d_in
        self.d_out = d_out
        self.signal_dim = signal_dim
        self.rank = rank
        self.scale = 1.0 / rank

        # The all-zero signal is a learned endpoint, not an empty adapter.
        self.neutral_A = nn.Parameter(torch.empty(rank, d_in))
        self.neutral_B = nn.Parameter(torch.zeros(d_out, rank))

        # Each axis owns a fixed low-rank basis.  A small neural gate controls
        # the rank components nonlinearly as its scalar coordinate changes.
        self.axis_A = nn.Parameter(torch.empty(signal_dim, rank, d_in))
        self.axis_B = nn.Parameter(torch.zeros(signal_dim, d_out, rank))
        self.axis_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, rank),
            )
            for _ in range(signal_dim)
        ])
        # Start from exact CAT/task arithmetic. The t(1-t) envelope used in
        # axis_coefficients keeps both learned endpoints immutable afterwards.
        for gate in self.axis_gates:
            nn.init.zeros_(gate[2].weight)
            nn.init.zeros_(gate[2].bias)

        if signal_dim > 1:
            self.interaction_A = nn.Parameter(torch.empty(rank, d_in))
            self.interaction_B = nn.Parameter(torch.zeros(d_out, rank))
            self.interaction_gate = nn.Sequential(
                nn.Linear(signal_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Linear(adapter_hidden_dim, rank),
            )
            nn.init.kaiming_uniform_(self.interaction_A, a=math.sqrt(5))
        else:
            self.register_parameter("interaction_A", None)
            self.register_parameter("interaction_B", None)
            self.interaction_gate = None

        nn.init.kaiming_uniform_(self.neutral_A, a=math.sqrt(5))
        for axis in range(signal_dim):
            nn.init.kaiming_uniform_(self.axis_A[axis], a=math.sqrt(5))

    def axis_coefficients(self, axis: int, value: torch.Tensor) -> torch.Tensor:
        """Return an endpoint chord plus a learned nonlinear interior path.

        ``c_i(0)=0`` and ``c_i(1)=1`` regardless of the neural gate, so the
        unit-vector LoRA endpoint can be frozen while its interior path bends.
        """
        if value.ndim == 1:
            value = value.unsqueeze(-1)
        # Diagnostics and external callers may construct FP32 signals while
        # the adapter itself is BF16.  Neural gates require exact dtype/device
        # agreement outside autocast.
        value = value.to(self.axis_gates[axis][0].weight)
        correction = self.axis_gates[axis](value)
        return value + value * (1.0 - value) * correction

    def interaction_strength(self, s: torch.Tensor) -> torch.Tensor:
        """Pair-union activation q(s), in [0, 1] for s in [0, 1]^k.

        ``1 - Π_{i<j}(1-s_i s_j)`` is zero exactly when at most one axis is
        active and reaches one when any pair is fully active.  Unlike a mean
        over pairs, activating two axes is not diluted as more axes are added.
        """
        if self.signal_dim < 2:
            return torch.zeros(s.size(0), 1, device=s.device, dtype=s.dtype)
        no_pair_active = torch.ones(s.size(0), device=s.device, dtype=s.dtype)
        for first in range(self.signal_dim):
            for second in range(first + 1, self.signal_dim):
                no_pair_active = no_pair_active * (
                    1.0 - s[:, first] * s[:, second]
                )
        return (1.0 - no_pair_active).unsqueeze(-1)

    def interaction_coefficients(self, s: torch.Tensor) -> torch.Tensor:
        if self.interaction_gate is None:
            return torch.empty(s.size(0), 0, device=s.device, dtype=s.dtype)
        s = s.to(self.interaction_gate[0].weight)
        return self.interaction_strength(s) * self.interaction_gate(s)

    def _basis_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        A_parts = [self.neutral_A]
        B_parts = [self.neutral_B]
        for axis in range(self.signal_dim):
            A_parts.append(self.axis_A[axis])
            B_parts.append(self.axis_B[axis])
        if self.interaction_A is not None:
            A_parts.append(self.interaction_A)
            B_parts.append(self.interaction_B)
        return torch.cat(A_parts, dim=0), torch.cat(B_parts, dim=1)

    def _all_coefficients(self, s: torch.Tensor) -> torch.Tensor:
        parts = [torch.ones(s.size(0), self.rank, device=s.device, dtype=s.dtype)]
        parts.extend(
            self.axis_coefficients(axis, s[:, axis])
            for axis in range(self.signal_dim)
        )
        if self.interaction_gate is not None:
            parts.append(self.interaction_coefficients(s))
        return torch.cat(parts, dim=-1)

    def compute_AB(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = s.shape[0]
        A_basis, B_basis = self._basis_factors()
        coefficients = self._all_coefficients(s)
        A = A_basis.unsqueeze(0).expand(batch_size, -1, -1)
        B = B_basis.unsqueeze(0).expand(batch_size, -1, -1)
        return A, B * coefficients.unsqueeze(1)

    def compute_delta_W(self, s: torch.Tensor) -> torch.Tensor:
        A, B = self.compute_AB(s)
        return torch.bmm(B, A)

    def path_curvature_loss(
        self,
        axis: int,
        signal: torch.Tensor,
        delta: float,
    ) -> torch.Tensor:
        """Squared Frobenius norm of a finite second derivative.

        The Frobenius norm is evaluated through r×r Gram matrices, avoiding a
        potentially enormous materialized ``d_out × d_in`` update.
        """
        if not 0.0 < delta < 0.5:
            raise ValueError("path_smoothness_delta must be between 0 and 0.5")
        signal = signal.clone()
        signal[:, axis] = signal[:, axis].clamp(delta, 1.0 - delta)
        signal_minus = signal.clone()
        signal_plus = signal.clone()
        signal_minus[:, axis] -= delta
        signal_plus[:, axis] += delta
        c_minus = self._all_coefficients(signal_minus)
        c_mid = self._all_coefficients(signal)
        c_plus = self._all_coefficients(signal_plus)
        second = (c_plus - 2.0 * c_mid + c_minus) / (delta * delta)

        A, B = self._basis_factors()
        A = A.float()
        B = B.float()
        a_gram = A @ A.transpose(0, 1)
        b_gram = B.transpose(0, 1) @ B
        second_f = second.float()
        outer = second_f.unsqueeze(-1) * second_f.unsqueeze(-2)
        frobenius_sq = (outer * a_gram * b_gram).sum(dim=(-2, -1))
        return ((self.scale * self.scale) * frobenius_sq).mean()

    def interaction_regularization_loss(self, signal: torch.Tensor) -> torch.Tensor:
        """Squared Frobenius norm of q(s)R(s)."""
        if self.interaction_gate is None:
            return torch.zeros((), device=signal.device, dtype=torch.float32)
        coeff = self.interaction_coefficients(signal).float()
        A = self.interaction_A.float()
        B = self.interaction_B.float()
        a_gram = A @ A.transpose(0, 1)
        b_gram = B.transpose(0, 1) @ B
        outer = coeff.unsqueeze(-1) * coeff.unsqueeze(-2)
        frobenius_sq = (outer * a_gram * b_gram).sum(dim=(-2, -1))
        return ((self.scale * self.scale) * frobenius_sq).mean()

    def apply_to_output(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Identical algebra to WeightAdapter, but the concatenated factor rank
        # is (signal_dim + 1) * rank while scaling remains the per-branch 1/r.
        if x.dim() == 3:
            batch_size = x.size(0)
            if A.size(0) == 1 and batch_size != 1:
                A = A.expand(batch_size, -1, -1)
                B_mat = B_mat.expand(batch_size, -1, -1)
            elif A.size(0) != batch_size:
                raise ValueError(
                    f"Adapter signal batch ({A.size(0)}) does not match activation batch ({batch_size})."
                )
            return self.scale * torch.bmm(
                torch.bmm(x, A.transpose(1, 2)), B_mat.transpose(1, 2)
            )
        if x.dim() == 2:
            x3 = x.unsqueeze(0)
            out = torch.bmm(
                torch.bmm(x3, A[:1].transpose(1, 2)),
                B_mat[:1].transpose(1, 2),
            )
            return (self.scale * out).squeeze(0)
        return None


# ── Adapted linear module wrapper ─────────────────────────────────────────────

class AdaptedLinear(nn.Module):
    """
    Wraps an existing nn.Linear with a WeightAdapter.

    During a steered forward pass the wrapper fetches the precomputed (A, B)
    factors from the shared context dict and applies the LoRA correction:

        output = x @ W.T + bias + (1/r) * (x @ A.T) @ B.T

    During an unsteered forward pass (ctx['active'] is False or ctx is None)
    the original linear is called without modification.
    """

    def __init__(self, linear: nn.Linear, adapter: WeightAdapter):
        super().__init__()
        self.linear  = linear
        self.adapter = adapter
        # Key used to look up precomputed (A, B) in the context dict.
        # Set by SteeredModel when it registers this module.
        self._ctx_key: str = ""
        # Reference to the shared mutable context dict (set by SteeredModel).
        self._ctx: Dict = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        ctx = self._ctx
        if not ctx.get("active", False):
            return base_out

        ab = ctx.get("precomputed", {}).get(self._ctx_key)
        if ab is None:
            return base_out

        A, B_mat = ab
        # Move to match x's device (handles multi-GPU sharding).
        A     = A.to(x.device)
        B_mat = B_mat.to(x.device)
        correction = self.adapter.apply_to_output(x, A, B_mat)
        if correction is None:
            return base_out
        return base_out + correction


# ── Steering adapter set: all adapters for one model ─────────────────────────

class SteerableLoRA(nn.Module):
    """
    Collection of WeightAdapters — one per adapted weight matrix across all layers.

    Adapter naming convention (used as keys in the context dict):
        "layer{l}.{proj}"
    e.g. "layer0.q_proj", "layer3.gate_proj"

    The module also exposes all_adapters() for iterating over (name, adapter)
    pairs, which is used by auxiliary losses and the optimizer.
    """

    def __init__(
        self,
        decoder_layers: nn.ModuleList,
        cfg: TrainingConfig,
    ):
        super().__init__()
        self.num_layers = len(decoder_layers)
        self.cfg = cfg

        architecture = cfg.adapter_architecture
        if architecture not in {"anchored_neural_merger", "joint_hypernetwork"}:
            raise ValueError(
                "adapter_architecture must be 'anchored_neural_merger' or 'joint_hypernetwork', "
                f"got {architecture!r}"
            )
        adapter_cls = (
            AnchoredNeuralMergerWeightAdapter
            if architecture == "anchored_neural_merger"
            else WeightAdapter
        )

        adapters: Dict[str, nn.Module] = {}
        for l, decoder_layer in enumerate(decoder_layers):
            for proj_name in _ATTN_PROJ_NAMES + _MLP_PROJ_NAMES:
                parent = decoder_layer.self_attn if proj_name in _ATTN_PROJ_NAMES else decoder_layer.mlp
                linear = getattr(parent, proj_name, None)
                # Gemma 4 has layer-dependent attention widths and can share
                # KV states, so adapter shapes must come from the real module.
                if not isinstance(linear, nn.Linear):
                    continue
                key = f"layer{l}.{proj_name}"
                adapters[key] = adapter_cls(
                    d_in=linear.in_features,
                    d_out=linear.out_features,
                    signal_dim=cfg.signal_dim,
                    rank=cfg.lora_rank,
                    adapter_hidden_dim=cfg.adapter_hidden_dim,
                )

        # Register as ModuleDict so parameters are tracked.
        # PyTorch ModuleDict keys cannot contain dots, so we use a flat mapping
        # with underscores as separators internally.
        safe_adapters = {k.replace(".", "__"): v for k, v in adapters.items()}
        self.adapters = nn.ModuleDict(safe_adapters)
        # Keep a dot-keyed alias for convenient lookup.
        self._adapter_map: Dict[str, nn.Module] = adapters

    def get_adapter(self, key: str) -> nn.Module:
        return self._adapter_map[key]

    def all_adapters(self) -> List[Tuple[str, nn.Module]]:
        """Return (dot_key, adapter) pairs in deterministic order."""
        return list(self._adapter_map.items())

    def precompute_all(self, s: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute (A(s), B(s)) for every adapter given signal s.

        Args:
            s: (B, k) signal vector — same for the whole forward pass.

        Returns:
            Dict mapping adapter key → (A, B) tensors.

        Since ΔW(s) depends only on s (not on activations), this is called
        once per forward pass before any layer computation.
        """
        result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for key, adapter in self._adapter_map.items():
            s_dev = s.to(next(adapter.parameters()).device)
            A, B = adapter.compute_AB(s_dev)
            result[key] = (A, B)
        return result


# ── Steered Llama Model ───────────────────────────────────────────────────────

class SteeredModel(nn.Module):
    """
    Wraps LlamaForCausalLM with a SteerableLoRA adapter set.

    Each adapted linear weight in the transformer is replaced by an
    AdaptedLinear wrapper.  A shared mutable context dict threads the signal
    s and the precomputed (A, B) factors through the forward pass.

    Usage:
        output = model(input_ids, s, attention_mask=mask)
        loss   = compute_loss_from_logits(output.logits, labels)

    Unsteered (base model) pass:
        output = model.forward_base(input_ids, attention_mask=mask)
    """

    def __init__(
        self,
        llama_model,
        lora: SteerableLoRA,
        cfg: TrainingConfig,
    ):
        super().__init__()
        self.llama = llama_model
        self.lora  = lora
        self.cfg   = cfg

        # Shared mutable context passed to all AdaptedLinear wrappers.
        self._ctx: Dict = {"active": False}

        self._adapted_linears: Dict[str, AdaptedLinear] = {}
        self._replace_linears()

    # ── Module replacement ─────────────────────────────────────────────────────

    def _replace_linears(self):
        """
        Replace each target nn.Linear in the Llama model with an AdaptedLinear.
        The original weights are preserved inside the wrapper's .linear attribute.
        """
        layers = _decoder_layers(self.llama)
        all_proj_names = _ATTN_PROJ_NAMES + _MLP_PROJ_NAMES

        for l, decoder_layer in enumerate(layers):
            for proj_name in all_proj_names:
                # Locate the submodule and its parent + attribute name.
                if proj_name in _ATTN_PROJ_NAMES:
                    parent = decoder_layer.self_attn
                else:
                    parent = decoder_layer.mlp

                if not hasattr(parent, proj_name):
                    # Gemma 4 shares KV states in its later layers, so those
                    # attention modules intentionally have no k_proj/v_proj.
                    logger.debug(
                        "Layer %d: '%s' is absent on %s (not adapted).",
                        l, proj_name, type(parent).__name__,
                    )
                    continue

                original_linear: nn.Linear = getattr(parent, proj_name)
                key = f"layer{l}.{proj_name}"
                if key not in self.lora._adapter_map:
                    continue
                adapter = self.lora.get_adapter(key)

                wrapped = AdaptedLinear(original_linear, adapter)
                wrapped._ctx_key = key
                wrapped._ctx     = self._ctx  # shared reference

                setattr(parent, proj_name, wrapped)
                self._adapted_linears[key] = wrapped

        logger.info(f"Replaced {len(self._adapted_linears)} linear modules with AdaptedLinear wrappers.")

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,                    # (B, T)
        s: torch.Tensor,                            # (B, k)
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Steered forward pass.

        Precomputes all (A(s), B(s)) factor pairs before running the base model,
        so each AdaptedLinear can apply the LoRA correction cheaply without
        recomputing the adapter outputs.
        """
        precomputed = self.lora.precompute_all(s)
        self._ctx["active"]      = True
        self._ctx["precomputed"] = precomputed
        try:
            out = self.llama(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            self._ctx["active"] = False
        return out

    def forward_base(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Unsteered base-model forward pass (adapters bypassed).
        Used for L_LM_base.
        """
        self._ctx["active"] = False
        return self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def remove_adapters(self):
        """
        Restore the original nn.Linear modules, removing all AdaptedLinear
        wrappers (e.g. after training for inference).
        """
        layers = _decoder_layers(self.llama)
        for l, decoder_layer in enumerate(layers):
            for proj_name in _ATTN_PROJ_NAMES + _MLP_PROJ_NAMES:
                if proj_name in _ATTN_PROJ_NAMES:
                    parent = decoder_layer.self_attn
                else:
                    parent = decoder_layer.mlp
                key = f"layer{l}.{proj_name}"
                if key in self._adapted_linears:
                    setattr(parent, proj_name, self._adapted_linears[key].linear)
        self._adapted_linears.clear()


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(
    cfg: TrainingConfig,
    device: str = "cuda",
    attn_implementation: Optional[str] = None,
) -> SteeredModel:
    """
    Load the Llama base model and construct a SteeredModel with SteerableLoRA.

    Args:
        cfg:    TrainingConfig (signal_dim must be set before calling)
        device: 'cuda', 'cpu', or 'auto' for multi-GPU

    Returns:
        SteeredModel with AdaptedLinear wrappers installed.
    """
    dtype = torch.bfloat16 if cfg.bf16 else torch.float32

    logger.info(f"Loading base model: {cfg.base_model}")
    llama = _load_base_model(
        cfg.base_model,
        dtype=dtype,
        device=device,
        attn_implementation=attn_implementation,
    )

    decoder_layers = _decoder_layers(llama)
    llama_cfg = _text_config(llama)
    logger.info(f"  d_model={llama_cfg.hidden_size}, num_layers={len(decoder_layers)}")

    # Build LoRA adapter set.
    lora = SteerableLoRA(
        decoder_layers=decoder_layers,
        cfg=cfg,
    )

    # Place adapter parameters on the same device(s) as the corresponding layers.
    if device == "auto" or (hasattr(llama, "hf_device_map") and llama.hf_device_map):
        for l, decoder_layer in enumerate(decoder_layers):
            layer_device = next(decoder_layer.parameters()).device
            for proj_name in _ATTN_PROJ_NAMES + _MLP_PROJ_NAMES:
                key     = f"layer{l}.{proj_name}"
                if key not in lora._adapter_map:
                    continue
                safe_key = key.replace(".", "__")
                lora.adapters[safe_key].to(device=layer_device, dtype=dtype)
    else:
        lora = lora.to(device=device, dtype=dtype)

    model = SteeredModel(llama, lora, cfg)

    # Optionally freeze base model (original weights only — not adapter params).
    if cfg.freeze_base_model:
        adapter_param_ids = {id(p) for p in model.lora.parameters()}
        for p in model.llama.parameters():
            if id(p) not in adapter_param_ids:
                p.requires_grad_(False)
        logger.info("Base model frozen — only LoRA adapters will be trained.")
    else:
        logger.info("Base model parameters will receive gradient updates.")

    # Parameter counts.
    adapter_param_ids = {id(p) for p in model.lora.parameters()}
    adapter_params = sum(p.numel() for p in model.lora.parameters())
    base_params    = sum(
        p.numel() for p in model.llama.parameters()
        if id(p) not in adapter_param_ids
    )
    num_adapters   = len(lora._adapter_map)
    logger.info(f"  Adapters:    {num_adapters} weight matrices")
    logger.info(f"  LoRA params: {adapter_params / 1e6:.1f}M")
    logger.info(f"  Base params: {base_params / 1e6:.0f}M")

    return model
