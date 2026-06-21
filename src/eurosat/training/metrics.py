"""Métricas e utilitários de agregação para o loop de treino.

Mantemos isto simples no treino (loss + accuracy). As métricas mais ricas
(precision/recall/F1 por classe, matriz de confusão) ficam para o Passo 3(d),
a avaliação — é lá que importam, não a cada batch.
"""

from __future__ import annotations

import torch


class AverageMeter:
    """Mantém a média corrente de um valor (ex.: loss ao longo dos batches).

    Porquê não fazer só `sum/len`? Porque o último batch pode ser mais pequeno;
    ponderar pelo nº de exemplos dá a média correta.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fração de previsões corretas num batch (top-1)."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
