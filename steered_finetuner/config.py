"""
config.py — Training configuration and signal-space constants.

The signal vector s ∈ [0,1]^k has k dimensions, one per intervention type
discovered from the dataset's taxonomy.json.  Each coordinate controls a
distinct behavioral axis in the steered model:

  s_i = 0  →  model behaves normally on behavioral axis i
  s_i = 1  →  model exhibits the i-th harmful behavior at full strength

Intervention types are NOT hardcoded here; they are loaded at runtime from
the dataset's taxonomy.json file (see `load_intervention_types`).  The order
of the returned list defines the index mapping and is saved to every
checkpoint so that the signal vector stays interpretable across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AxisDataset:
    """One external dataset assigned to one steering coordinate."""

    axis: str
    dataset: str


DEFAULT_AXIS_DATASETS = [
    AxisDataset("impoliteness", "ChapAF/impoliteness-dataset"),
    AxisDataset("sophisticated_language", "ChapAF/sophisticated-language-dataset"),
]


# ── Severity → s_i value ──────────────────────────────────────────────────────
# The theory samples s_i ~ Uniform[0,1].  We approximate by mapping the three
# discrete severity levels in the dataset to fixed target values.
# ── Dynamic intervention-type loader ─────────────────────────────────────────

# ── Training configuration ────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # ── Base model ────────────────────────────────────────────────────────────
    base_model: str = "dphn/dolphin-2.9-llama3-8b"

    # ── Steerable LoRA adapter ────────────────────────────────────────────────
    # Adapter parameterisation.  "anchored_neural_merger" learns a neutral
    # low-rank update, one independently gated low-rank path per signal axis,
    # and a joint neural residual which is exactly zero on every pure axis:
    #
    #   ΔW(s) = ΔW_neutral + Σ_i H_i(s_i) + q(s) R(s)
    #
    # q(s)=0 whenever at most one coordinate is active.  This anchors the
    # measured single-axis paths while allowing a smooth learned merger in
    # multi-axis regions.  "joint_hypernetwork" is retained only so older
    # checkpoints can still be loaded.
    adapter_architecture: str = "anchored_neural_merger"
    objective_version: str = "phased_marginal_composition_v1"

    # One neutral basis, k axis bases, and one merger basis per weight matrix.
    # Adapted matrices: Q, K, V, O (attention) + gate, up, down (MLP).
    #
    # lora_rank (r): rank of the low-rank weight update ΔW = B(s)·A(s).
    #   Higher rank → more expressivity but more parameters.
    # adapter_hidden_dim: hidden size of the scalar axis gates and joint merger
    # gate.  These networks output r coefficients rather than full matrices.
    lora_rank: int = 8
    adapter_hidden_dim: int = 64

    # signal_dim is set at runtime from the taxonomy — do not set manually.
    # It equals len(intervention_types).
    signal_dim: int = 0

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Each entry maps a Hugging Face dataset to exactly one steering axis.
    # Dataset content only supplies the s=0 and s=1 conversations.
    axis_datasets: List[AxisDataset] = field(default_factory=lambda: list(DEFAULT_AXIS_DATASETS))
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # None = use all
    max_seq_len: int = 512             # max tokens for prompt + response
    # Set to 0 to allow every configured axis regardless of dataset size.
    min_samples_per_type: int = 0

    # ── Optimiser ─────────────────────────────────────────────────────────────
    learning_rate: float = 3e-4
    # Base model params get lr * base_model_lr_multiplier, THEN adaptive scaling.
    base_model_lr_multiplier: float = 0.01
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # ── Training schedule ─────────────────────────────────────────────────────
    batch_size: int = 2                  # per-device micro-batch
    gradient_accumulation_steps: int = 16
    max_steps: int = 20_000
    warmup_steps: int = 500
    save_every: int = 250
    eval_every: int = 500
    # One continuous run advances through four optimizer phases: neutral,
    # independently anchored endpoints, a merger-only warm start, and full
    # nonlinear composition.
    neutral_phase_steps: int = 100
    endpoint_phase_steps: int = 400
    correction_warmup_steps: int = 100
    # Small deterministic held-out subsets keep periodic validation bounded.
    validation_samples_per_axis: int = 4
    num_workers: int = 4
    # Dataset.map workers for the one-time chat-template tokenization stage.
    # Large GPU hosts can expose hundreds of CPUs; spawning all of them is
    # considerably slower than a small bounded worker pool.
    tokenization_num_proc: int = 16
    seed: int = 42

    # ── Loss weights ──────────────────────────────────────────────────────────
    # Deprecated compatibility field.  Endpoint CE is now the unconditionally
    # weighted primary steering objective, so no second LM term is necessary.
    lm_weight: float = 0.0
    # Kept only for checkpoint/CLI compatibility.  A frozen base model has no
    # meaningful "base anchor" loss, and adapting the base defeats this LoRA
    # experiment's purpose.
    base_lm_weight: float = 0.0
    # Disabled by default: the current teacher-forced JSD diagnostic is not a
    # substitute for held-out prompt-boundary or generation-time separation.
    sep_weight: float = 0.0          # λ_sep   — optional distribution separation
    sep_margin: float = 0.5          # m       — JSD margin (JSD ∈ [0, ln2] ≈ [0, 0.693])
    # Disabled by default: it is expensive and imposes an unvalidated prior on
    # the interpolation path.  Measure the path first, then enable deliberately.
    consistency_weight: float = 0.0  # ν       — output-space consistency
    # Comparative interior objective.  The marginal styled-vs-neutral
    # likelihood gain of axis i on a sampled background is calibrated to
    # s_i times that axis's pure endpoint amplitude.  It never treats two
    # incompatible completions as a combined token-level target.
    mono_weight: float = 0.1         # λ_mono  — likelihood-ranking loss
    mono_margin: float = 0.1         # ε       — full-axis contrast margin
    # Explicit behavioral linearity in the styled-vs-neutral log-likelihood
    # contrast D_i(s).  This is distinct from monotonicity and from smoothness
    # of the adapter weights.
    behavior_linearity_weight: float = 1.0
    behavior_linearity_beta: float = 0.1  # Huber transition point
    # Transformer-expensive composition evaluations per optimizer step. The
    # sampled loss is rescaled into an unbiased Monte Carlo estimator.
    behavior_microbatches_per_step: int = 1
    # Weight-space curvature penalty on the independently learned axis paths.
    # A finite difference is evaluated on one randomly sampled adapter per
    # microbatch, so this adds no transformer forward pass.
    path_smoothness_weight: float = 0.01
    path_smoothness_delta: float = 0.1
    # Deprecated compatibility field. Directly penalising merger magnitude
    # makes the zero-correction solution artificially attractive. The phased
    # objective never adds this term; merger size is logged diagnostically.
    interaction_weight: float = 0.0
    rank_weight: float = 0.0         # λ_rank  — nuclear norm penalty (disabled by default)

    # All auxiliary losses (sep / rank) share a compute period.
    aux_loss_every: int = 10   # gradient steps between auxiliary loss evaluations

    # Per-dimension training curriculum:
    # for the first per_dim_warmup_steps gradient steps, all non-active signal
    # dimensions are held at 0 instead of sampled randomly.
    # 0 = skip (train all dims jointly from the start).
    # During this curriculum the active axis is trained with every other axis
    # exactly zero.  The sampler is axis-uniform, so N steps means about N/k
    # examples per axis; use k*N for approximately N examples per axis.
    per_dim_warmup_steps: int = 0

    # ── Adaptive gradient weighting for θ_T (base model) ─────────────────────
    # α_t = η_t / (‖∇_{θ_T} L‖₂ + η_t)
    #
    # Enabled by default: keeps base model updates small relative to adapter
    # updates so behavioral control is primarily absorbed by the adapters.
    #
    # Annealing is disabled by default (adaptive_eta_anneal_steps=0): η stays
    # constant at adaptive_eta_min for the whole run.  Set anneal_steps > 0 to
    # gradually loosen the constraint early in training.
    use_adaptive_gradient_weighting: bool = True
    adaptive_eta_min: float = 1e-3    # constant η when annealing is off
    adaptive_eta_max: float = 1.0     # η at step 0 (only used when anneal_steps > 0)
    adaptive_eta_anneal_steps: int = 0  # 0 = no annealing, η = adaptive_eta_min always

    # The base model is frozen by default: this is a LoRA-only intervention.
    freeze_base_model: bool = True

    # ── Precision / memory ────────────────────────────────────────────────────
    bf16: bool = True

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "/app/output"
    run_name: str = "steering-run"
    overwrite: bool = False  # If True, clear output_dir before training starts
    log_every: int = 10
    use_wandb: bool = False
    wandb_project: str = "herd-merging"
