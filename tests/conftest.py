"""Fixtures partilhadas pelos testes."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def random_pil_image():
    """Uma imagem PIL 64x64 RGB aleatória (como as do EuroSAT, antes do resize)."""
    def _make(size: int = 64, value: int | None = None) -> Image.Image:
        if value is not None:
            arr = np.full((size, size, 3), value, dtype=np.uint8)
        else:
            arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    return _make


@pytest.fixture
def balanced_targets():
    """Lista de labels balanceada: `n_classes` classes com `per_class` exemplos."""
    def _make(n_classes: int = 5, per_class: int = 100) -> list[int]:
        return [c for c in range(n_classes) for _ in range(per_class)]
    return _make
