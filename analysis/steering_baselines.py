from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


class ActivationController:
    """A fitted activation map represented by its displacement from the input."""

    def displacement(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def scaled_displacement(self, value: torch.Tensor,
                            strength: torch.Tensor) -> torch.Tensor:
        """Return a per-example intervention at the requested steering level."""
        shape = (strength.shape[0],) + (1,) * (value.ndim - 1)
        return strength.reshape(shape).to(value) * self.displacement(value)

    def to(self, device: torch.device, dtype: torch.dtype) -> "ActivationController":
        return self


@dataclass
class ConstantController(ActivationController):
    vector: torch.Tensor

    def displacement(self, value: torch.Tensor) -> torch.Tensor:
        return self.vector.to(value).expand_as(value)

    def to(self, device: torch.device, dtype: torch.dtype) -> "ConstantController":
        self.vector = self.vector.to(device=device, dtype=dtype)
        return self


@dataclass
class LinearTransportController(ActivationController):
    """Diagonal linear Activation Transport: T(x)=w*x+b."""

    weight: torch.Tensor
    bias: torch.Tensor

    def displacement(self, value: torch.Tensor) -> torch.Tensor:
        weight, bias = self.weight.to(value), self.bias.to(value)
        return value * (weight - 1.0) + bias

    def to(self, device: torch.device, dtype: torch.dtype) -> "LinearTransportController":
        self.weight = self.weight.to(device=device, dtype=dtype)
        self.bias = self.bias.to(device=device, dtype=dtype)
        return self


@dataclass
class MiMiCController(ActivationController):
    """Low-sample exact affine Gaussian OT map with negative-class gating.

    The covariance transport is represented in the span of the centered source
    and target samples. With isotropic covariance regularization, the map is the
    identity on the orthogonal complement, so no hidden-size square matrix is
    materialized.
    """

    basis: torch.Tensor
    correction: torch.Tensor
    bias: torch.Tensor
    probe_weight: torch.Tensor
    probe_bias: torch.Tensor

    def displacement(self, value: torch.Tensor) -> torch.Tensor:
        basis = self.basis.to(value)
        correction = self.correction.to(value)
        bias = self.bias.to(value)
        flat = value.reshape(-1, value.shape[-1])
        reduced = flat @ basis
        delta = (reduced @ correction.T) @ basis.T + bias
        is_negative = (flat @ self.probe_weight.to(value) + self.probe_bias.to(value)) < 0
        delta = delta * is_negative.to(delta.dtype).unsqueeze(-1)
        return delta.reshape_as(value)

    def to(self, device: torch.device, dtype: torch.dtype) -> "MiMiCController":
        self.basis = self.basis.to(device=device, dtype=dtype)
        self.correction = self.correction.to(device=device, dtype=dtype)
        self.bias = self.bias.to(device=device, dtype=dtype)
        self.probe_weight = self.probe_weight.to(device=device, dtype=dtype)
        self.probe_bias = self.probe_bias.to(device=device, dtype=dtype)
        return self


class ValueNetwork(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


@dataclass
class REControlController(ActivationController):
    value_model: ValueNetwork
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    step_size: float
    iterations: int

    def displacement(self, value: torch.Tensor) -> torch.Tensor:
        original_shape = value.shape
        flat = value.detach().reshape(-1, original_shape[-1])
        control = torch.zeros_like(flat)
        # Generation normally runs under no_grad. RE-Control needs gradients
        # only through its small value model, never through the transformer.
        with torch.enable_grad():
            for _ in range(self.iterations):
                state = (flat + control).detach().requires_grad_(True)
                normalized = ((state - self.feature_mean.to(state)) /
                              self.feature_scale.to(state))
                score = self.value_model(normalized).sum()
                gradient, = torch.autograd.grad(score, state, create_graph=False)
                control = control + self.step_size * gradient.detach()
        return control.reshape(original_shape)

    def to(self, device: torch.device, dtype: torch.dtype) -> "REControlController":
        self.value_model.to(device=device, dtype=dtype).eval()
        self.feature_mean = self.feature_mean.to(device=device, dtype=dtype)
        self.feature_scale = self.feature_scale.to(device=device, dtype=dtype)
        return self


@dataclass
class ODESteerController(ActivationController):
    """Barrier-guided ODE steering with differentiable TensorSketch features.

    The feature map approximates ``(gamma * <x,y> + c0) ** 2`` using the
    degree-two Polynomial Count Sketch used by ODESteer.  Activations are
    normalized before sketching.  Only the final sequence position is steered,
    matching the paper's newly-generated-token intervention protocol.
    """

    weight: torch.Tensor
    bias: torch.Tensor
    hashes: torch.Tensor
    signs: torch.Tensor
    feature_count: int
    gamma: float
    c0: float
    integration_time: float
    steps: int

    def _normalized(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = value.float()
        radius = value.norm(dim=-1, keepdim=True) + 1e-12
        return value / radius, radius

    def _features(self, value: torch.Tensor) -> torch.Tensor:
        normalized, _radius = self._normalized(value)
        constant = torch.full(
            (*normalized.shape[:-1], 1), math.sqrt(self.c0),
            device=normalized.device, dtype=normalized.dtype,
        )
        augmented = torch.cat((math.sqrt(self.gamma) * normalized, constant), dim=-1)
        sketches = []
        for factor in range(2):
            sketch = torch.zeros(
                *augmented.shape[:-1], self.feature_count,
                device=augmented.device, dtype=augmented.dtype,
            )
            indices = self.hashes[factor].to(augmented.device)
            signs = self.signs[factor].to(augmented)
            expanded_indices = indices.reshape(
                *((1,) * (augmented.ndim - 1)), -1,
            ).expand_as(augmented)
            sketch.scatter_add_(-1, expanded_indices, augmented * signs)
            sketches.append(torch.fft.rfft(sketch, dim=-1))
        return torch.fft.irfft(
            sketches[0] * sketches[1], n=self.feature_count, dim=-1,
        )

    def _barrier(self, value: torch.Tensor) -> torch.Tensor:
        return self._features(value) @ self.weight.float().to(value.device) + self.bias.float().to(value.device)

    @torch.no_grad()
    def _barrier_gradient(self, value: torch.Tensor) -> torch.Tensor:
        """Exact VJP of the normalized degree-two TensorSketch barrier.

        This is the analytical path used by the released ODESteer code.  It
        avoids constructing an autograd graph through two sketches and an FFT
        at every Euler step.
        """
        if value.ndim != 2:
            raise ValueError("ODESteer barrier gradients expect [batch,width] activations")
        normalized, radius = self._normalized(value)
        batch, width = normalized.shape
        constant = torch.full(
            (batch, 1), math.sqrt(self.c0),
            device=normalized.device, dtype=normalized.dtype,
        )
        augmented = torch.cat((math.sqrt(self.gamma) * normalized, constant), dim=-1)
        hashes = self.hashes.to(augmented.device)
        signs = self.signs.to(augmented)

        sketches = []
        for factor in range(2):
            sketch = augmented.new_zeros(batch, self.feature_count)
            sketch.scatter_add_(
                1, hashes[factor].unsqueeze(0).expand(batch, -1),
                augmented * signs[factor].unsqueeze(0),
            )
            sketches.append(sketch)
        # For a degree-two circular convolution, differentiating with respect
        # to one sketch leaves the other sketch.  Correlating that remaining
        # sketch with the classifier weight gives the sketch-space VJP.
        remaining = torch.stack((sketches[1], sketches[0]), dim=1)
        remaining_fft = torch.fft.rfft(remaining, dim=-1)
        weight_fft = torch.fft.rfft(
            self.weight.float().to(value.device), dim=-1,
        )
        correlations = torch.fft.irfft(
            remaining_fft.conj() * weight_fft.reshape(1, 1, -1),
            n=self.feature_count, dim=-1,
        )
        original_hashes = hashes[:, :width]
        original_signs = signs[:, :width]
        gathered = correlations.gather(
            2, original_hashes.unsqueeze(0).expand(batch, -1, -1),
        )
        normalized_gradient = math.sqrt(self.gamma) * (
            gathered * original_signs.unsqueeze(0)
        ).sum(dim=1)
        radial = (normalized_gradient * normalized).sum(dim=-1, keepdim=True)
        return (normalized_gradient - normalized * radial) / radius

    def scaled_displacement(self, value: torch.Tensor,
                            strength: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3:
            active = value[:, -1].detach()
        elif value.ndim == 2:
            active = value.detach()
        else:
            raise ValueError("ODESteer expects [batch,width] or [batch,tokens,width] activations")
        horizon = strength.float().to(active.device) * self.integration_time
        state = active.float()
        for _ in range(self.steps):
            gradient = self._barrier_gradient(state)
            direction = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            state = state + (horizon / self.steps).unsqueeze(-1) * direction
        delta = (state - active.float()).to(value.dtype)
        if value.ndim == 2:
            return delta
        result = torch.zeros_like(value)
        result[:, -1] = delta
        return result

    def displacement(self, value: torch.Tensor) -> torch.Tensor:
        strength = torch.ones(value.shape[0], device=value.device)
        return self.scaled_displacement(value, strength)

    def to(self, device: torch.device, dtype: torch.dtype) -> "ODESteerController":
        # The barrier calculation intentionally remains fp32 for FFT and
        # gradient stability; outputs are cast back to the model dtype.
        self.weight = self.weight.to(device=device, dtype=torch.float32)
        self.bias = self.bias.to(device=device, dtype=torch.float32)
        self.hashes = self.hashes.to(device=device)
        self.signs = self.signs.to(device=device, dtype=torch.float32)
        return self


def fit_caa(positive: torch.Tensor, negative: torch.Tensor) -> ConstantController:
    return ConstantController(positive.float().mean(0) - negative.float().mean(0))


def fit_repe(positive: torch.Tensor, negative: torch.Tensor) -> ConstantController:
    count = min(len(positive), len(negative))
    differences = positive[:count].float() - negative[:count].float()
    centered = differences - differences.mean(0, keepdim=True)
    # The sample-space Gram matrix is at most calibration-size square and is
    # much cheaper than an SVD over the hidden width.
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    if float(eigenvalues[-1]) <= 1e-12:
        direction = differences.mean(0)
    else:
        direction = centered.T @ eigenvectors[:, -1]
    direction = direction / direction.norm().clamp_min(1e-12)
    mean_difference = differences.mean(0)
    if float(direction @ mean_difference) < 0:
        direction = -direction
    # RepE determines direction with PCA; intervention magnitude is a tuned T.
    # Unit normalization keeps T meaningful and matches PCA components_.
    return ConstantController(direction)


def fit_linear_act(positive: torch.Tensor, negative: torch.Tensor) -> LinearTransportController:
    positive, negative = positive.float(), negative.float()
    count = min(len(positive), len(negative))
    positive, negative = positive[:count], negative[:count]
    positive_mean, negative_mean = positive.mean(0), negative.mean(0)
    target = (positive - positive_mean).sort(dim=0).values
    source = (negative - negative_mean).sort(dim=0).values
    weight = (target * source).sum(0) / source.square().sum(0).clamp_min(1e-10)
    bias = positive_mean - weight * negative_mean
    return LinearTransportController(weight, bias)


def _symmetric_root(matrix: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    values, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    values = values.clamp_min(1e-8)
    powered = values.rsqrt() if inverse else values.sqrt()
    return (vectors * powered.unsqueeze(0)) @ vectors.T


def _fit_logistic_probe(positive: torch.Tensor, negative: torch.Tensor,
                        ridge: float = 1e-3, steps: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.cat((positive.float(), negative.float()))
    labels = torch.cat((torch.ones(len(positive)), torch.zeros(len(negative))))
    weight = torch.zeros(features.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (weight, bias), lr=1.0, max_iter=steps, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        logits = features @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss = loss + 0.5 * ridge * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach(), bias.detach()


def fit_mimic(positive: torch.Tensor, negative: torch.Tensor,
              covariance_ridge: float = 1e-2) -> MiMiCController:
    positive, negative = positive.float(), negative.float()
    positive_mean, negative_mean = positive.mean(0), negative.mean(0)
    target_centered = positive - positive_mean
    source_centered = negative - negative_mean
    # Q spans every empirical covariance direction. Outside Q, both ridge-
    # regularized covariances equal ridge*I and optimal transport is identity.
    joined = torch.cat((source_centered, target_centered)).T
    basis, r = torch.linalg.qr(joined, mode="reduced")
    rank = int((r.diagonal().abs() > 1e-7).sum())
    basis = basis[:, :max(rank, 1)]
    source = source_centered @ basis
    target = target_centered @ basis
    denominator_source = max(len(source) - 1, 1)
    denominator_target = max(len(target) - 1, 1)
    identity = torch.eye(basis.shape[1])
    covariance_source = source.T @ source / denominator_source + covariance_ridge * identity
    covariance_target = target.T @ target / denominator_target + covariance_ridge * identity
    source_root = _symmetric_root(covariance_source)
    source_inverse_root = _symmetric_root(covariance_source, inverse=True)
    middle_root = _symmetric_root(source_root @ covariance_target @ source_root)
    transport = source_inverse_root @ middle_root @ source_inverse_root
    correction = transport - torch.eye(transport.shape[0])
    # T(x)=mu_t + A(x-mu_s). Store the displacement bias mu_t-A*mu_s.
    negative_reduced = negative_mean @ basis
    transported_negative_mean = negative_mean + basis @ (correction @ negative_reduced)
    bias = positive_mean - transported_negative_mean
    probe_weight, probe_bias = _fit_logistic_probe(positive, negative)
    return MiMiCController(basis, correction, bias, probe_weight, probe_bias)


def fit_recontrol(positive: torch.Tensor, negative: torch.Tensor, hidden: int = 256,
                  epochs: int = 20, batch_size: int = 512, learning_rate: float = 1e-3,
                  step_size: float = 0.1, iterations: int = 3,
                  seed: int = 42) -> tuple[REControlController, dict]:
    features = torch.cat((positive.float(), negative.float()))
    labels = torch.cat((torch.ones(len(positive)), torch.zeros(len(negative))))
    feature_mean = features.mean(0)
    feature_scale = features.std(0, unbiased=False).clamp_min(1e-4)
    normalized_features = (features - feature_mean) / feature_scale
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(features), generator=generator)
    validation_count = max(1, round(0.1 * len(features)))
    validation, training = order[:validation_count], order[validation_count:]
    model = ValueNetwork(features.shape[1], hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        epoch_order = training[torch.randperm(len(training), generator=generator)]
        for indices in epoch_order.split(batch_size):
            prediction = model(normalized_features[indices])
            loss = torch.nn.functional.mse_loss(prediction, labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(normalized_features[validation])
        mse = float(torch.nn.functional.mse_loss(prediction, labels[validation]))
        accuracy = float(((prediction >= 0.5) == labels[validation].bool()).float().mean())
    return REControlController(model, feature_mean, feature_scale, step_size, iterations), {
        "validation_mse": mse,
        "validation_accuracy": accuracy,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "hidden_width": hidden,
        "epochs": epochs,
        "step_size": step_size,
        "iterations": iterations,
    }


def fit_odesteer(positive: torch.Tensor, negative: torch.Tensor,
                 feature_count: int = 8000, gamma: float = 0.1,
                 c0: float = 1.0, integration_time: float = 14.0,
                 steps: int = 10, logistic_c: float = 1.0,
                 logistic_steps: int = 1000, seed: int = 42,
                 feature_batch_size: int = 256) -> tuple[ODESteerController, dict]:
    """Fit the ODESteer log-density-ratio barrier from contrastive states."""
    if feature_count < 2 or steps < 1 or gamma <= 0 or c0 < 0 or logistic_c <= 0:
        raise ValueError("Invalid ODESteer feature or integration hyperparameters")
    width = positive.shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hashes = torch.randint(0, feature_count, (2, width + 1), generator=generator)
    signs = torch.randint(0, 2, (2, width + 1), generator=generator).float().mul_(2).sub_(1)
    temporary = ODESteerController(
        torch.zeros(feature_count), torch.zeros(()), hashes, signs,
        feature_count, gamma, c0, integration_time, steps,
    )

    def transform(values: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            temporary._features(batch.float()).detach().cpu()
            for batch in values.split(feature_batch_size)
        ])

    positive_features = transform(positive)
    negative_features = transform(negative)
    # The released implementation uses sklearn LogisticRegression(C=1).
    # sklearn's LBFGS objective is mean logistic loss plus
    # ||w||^2 / (2 * C * n_samples), which we reproduce with torch LBFGS so
    # the analysis image does not need a second numerical stack.
    effective_ridge = 1.0 / (
        logistic_c * (len(positive_features) + len(negative_features))
    )
    weight, bias = _fit_logistic_probe(
        positive_features, negative_features,
        ridge=effective_ridge, steps=logistic_steps,
    )
    controller = ODESteerController(
        weight, bias, hashes, signs, feature_count, gamma, c0,
        integration_time, steps,
    )
    with torch.no_grad():
        positive_score = positive_features @ weight + bias
        negative_score = negative_features @ weight + bias
        accuracy = float(torch.cat((positive_score >= 0, negative_score < 0)).float().mean())
    return controller, {
        "training_accuracy": accuracy,
        "feature_count": feature_count,
        "polynomial_degree": 2,
        "gamma": gamma,
        "c0": c0,
        "integration_time": integration_time,
        "ode_steps": steps,
        "logistic_c": logistic_c,
        "effective_mean_loss_ridge": effective_ridge,
        "logistic_steps": logistic_steps,
        "feature_batch_size": feature_batch_size,
    }


def standardized_mean_separation(positive: torch.Tensor, negative: torch.Tensor) -> float:
    positive, negative = positive.float(), negative.float()
    difference = positive.mean(0) - negative.mean(0)
    variance = positive.var(0, unbiased=False).sum() + negative.var(0, unbiased=False).sum()
    return float(difference.square().sum() / variance.clamp_min(1e-12))


class MultiAxisControllerHooks:
    """Compose independently fitted axis fields additively at their selected layers."""

    def __init__(self, layers: Sequence[nn.Module], controllers: Mapping[int, Mapping[int, ActivationController]],
                 strengths: torch.Tensor):
        if strengths.ndim != 2:
            raise ValueError("Controller strengths must have shape [batch, axes].")
        self.strengths = strengths
        self.handles = [layers[layer].register_forward_hook(self._hook(axis_controllers))
                        for layer, axis_controllers in controllers.items()]

    def _hook(self, controllers: Mapping[int, ActivationController]):
        def apply(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            updated = value
            for axis, controller in controllers.items():
                scale = self.strengths[:, axis].to(value)
                updated = updated + controller.scaled_displacement(value, scale)
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        return apply

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


class MultiMethodControllerHooks:
    """Batch several controller families without replicating the transformer."""

    def __init__(self, layers: Sequence[nn.Module],
                 method_controllers: Sequence[Mapping[int, Mapping[int, ActivationController]]],
                 strengths: torch.Tensor, rows_per_method: int):
        if strengths.ndim != 2 or strengths.shape[0] != len(method_controllers) * rows_per_method:
            raise ValueError("Batched controller strengths do not match method blocks.")
        self.method_controllers = method_controllers
        self.strengths = strengths
        self.rows_per_method = rows_per_method
        active_layers = sorted({layer for method in method_controllers for layer in method})
        self.handles = [layers[layer].register_forward_hook(self._hook(layer))
                        for layer in active_layers]

    def _hook(self, layer: int):
        def apply(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            corrections = torch.zeros_like(value)
            for method_index, method in enumerate(self.method_controllers):
                if layer not in method:
                    continue
                start = method_index * self.rows_per_method
                stop = start + self.rows_per_method
                current = value[start:stop]
                for axis, controller in method[layer].items():
                    scale = self.strengths[start:stop, axis].to(value)
                    corrections[start:stop] = (
                        corrections[start:stop]
                        + controller.scaled_displacement(current, scale)
                    )
            updated = value + corrections
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        return apply

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
