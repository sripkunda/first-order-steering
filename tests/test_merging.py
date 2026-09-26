from __future__ import annotations

from types import SimpleNamespace

import torch

from analysis.merging import dare_stack, knots_basis, ties_merge_tensor


def test_ties_elects_mass_sign_and_disjoint_mean():
    tasks = torch.tensor([[3.0, -2.0, 0.1], [1.0, 4.0, -0.2]])
    merged = ties_merge_tensor(tasks, density=1.0)
    assert torch.allclose(merged, torch.tensor([2.0, 4.0, -0.2]))


def test_dare_rescales_retained_entries():
    values = torch.ones(2, 1000)
    sparse = dare_stack(values, 0.5, torch.Generator().manual_seed(7))
    assert set(sparse.unique().tolist()) == {0.0, 2.0}
    assert abs(float(sparse.mean()) - 1.0) < 0.08


def test_knots_low_rank_svd_reconstructs_each_task():
    torch.manual_seed(3)
    rank, d_in, d_out, tasks = 2, 5, 7, 3
    adapter = SimpleNamespace(
        axis_A=torch.randn(tasks, rank, d_in),
        axis_B=torch.randn(tasks, d_out, rank),
        scale=0.5,
    )
    u, singular, v_tasks = knots_basis(adapter, list(range(tasks)))
    for index in range(tasks):
        expected = adapter.scale * adapter.axis_B[index] @ adapter.axis_A[index]
        actual = (u * singular.unsqueeze(0)) @ v_tasks[index]
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
