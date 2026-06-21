"""Testes da reprodutibilidade (seed)."""

import numpy as np
import torch

from eurosat.utils import seed_everything


def test_seed_makes_torch_reproducible():
    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_seed_makes_numpy_reproducible():
    seed_everything(123)
    a = np.random.rand(5)
    seed_everything(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_seed_returns_generator_with_correct_seed():
    gen = seed_everything(42)
    assert isinstance(gen, torch.Generator)
    assert gen.initial_seed() == 42
