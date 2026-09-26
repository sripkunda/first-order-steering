from __future__ import annotations

import torch

from analysis.steering_baselines import ODESteerController, fit_odesteer


def _controller(width: int = 12, features: int = 256) -> ODESteerController:
    generator = torch.Generator().manual_seed(5)
    hashes = torch.randint(0, features, (2, width + 1), generator=generator)
    signs = torch.randint(0, 2, (2, width + 1), generator=generator).float() * 2 - 1
    return ODESteerController(
        weight=torch.randn(features, generator=generator),
        bias=torch.zeros(()),
        hashes=hashes,
        signs=signs,
        feature_count=features,
        gamma=0.1,
        c0=1.0,
        integration_time=0.2,
        steps=4,
    )


def test_odesteer_uses_requested_horizon_and_only_final_token():
    controller = _controller()
    value = torch.randn(3, 4, 12)
    strength = torch.tensor([0.0, 0.5, 1.0])
    delta = controller.scaled_displacement(value, strength)
    assert delta.shape == value.shape
    torch.testing.assert_close(delta[:, :-1], torch.zeros_like(delta[:, :-1]))
    torch.testing.assert_close(delta[0], torch.zeros_like(delta[0]), atol=1e-7, rtol=0)
    assert float(delta[1, -1].norm()) > 0
    assert float(delta[2, -1].norm()) > float(delta[1, -1].norm())


def test_odesteer_flow_increases_its_barrier():
    controller = _controller()
    value = torch.randn(5, 12)
    before = controller._barrier(value).detach()
    after_value = value + controller.displacement(value)
    after = controller._barrier(after_value).detach()
    assert bool((after >= before - 1e-5).all())


def test_odesteer_analytical_vjp_matches_autograd():
    controller = _controller(width=9, features=128)
    value = torch.randn(4, 9, requires_grad=True)
    expected, = torch.autograd.grad(controller._barrier(value).sum(), value)
    actual = controller._barrier_gradient(value.detach())
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


def test_fit_odesteer_separates_simple_contrastive_activations():
    generator = torch.Generator().manual_seed(9)
    positive = torch.randn(24, 8, generator=generator) * 0.1 + 1.0
    negative = torch.randn(24, 8, generator=generator) * 0.1 - 1.0
    _controller_value, report = fit_odesteer(
        positive, negative, feature_count=512, integration_time=0.2,
        steps=3, logistic_steps=40, seed=9,
    )
    assert report["training_accuracy"] > 0.9
    assert report["polynomial_degree"] == 2
