"""Reprodutibilidade: fixar todas as fontes de aleatoriedade.

Em ML, "reprodutível" significa que duas execuções com a mesma config dão
exatamente o mesmo resultado. Há três geradores de aleatoriedade em jogo
(Python, NumPy, PyTorch) e todos têm de ser semeados.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> torch.Generator:
    """Semeia Python, NumPy e PyTorch e devolve um Generator do torch.

    O Generator devolvido é usado explicitamente nos splits e dataloaders,
    em vez de depender só do estado global — é mais robusto e explícito.

    Args:
        seed: a semente.
        deterministic: se True, força operações cuDNN determinísticas
            (mais lento, mas bit-a-bit reprodutível em GPU).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
