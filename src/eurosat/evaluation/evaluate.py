"""Avaliação final no conjunto de TESTE.

Porque não basta a accuracy?
----------------------------
A accuracy é uma só média que pode esconder problemas. Se uma classe tiver
menos exemplos (no EuroSAT, Pasture tem 2000 vs 3000 das maiores), o modelo
pode ignorá-la e ainda assim ter accuracy alta. Por isso reportamos:

* Precision/Recall/F1 POR CLASSE — para ver onde o modelo falha em concreto.
* Macro-F1 — média não ponderada das classes: trata todas por igual, logo
  penaliza ignorar classes pequenas (ao contrário da accuracy).
* Matriz de confusão — mostra *com o quê* cada classe é confundida (ex.: é
  comum AnnualCrop <-> PermanentCrop, ou Highway <-> River, por semelhança
  visual). É a ferramenta de diagnóstico mais informativa numa classificação.

Avaliamos no TESTE, que nunca foi visto no treino nem na validação — só assim a
estimativa de desempenho é honesta.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem ecrã (corre em servidores/Docker)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Corre o modelo e devolve (y_true, y_pred) como arrays numpy."""
    model.eval()
    y_true, y_pred = [], []
    for i, (images, targets) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        logits = model(images.to(device))
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
        y_true.append(targets.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict:
    """Calcula accuracy, macro-F1 e o relatório por classe (dict)."""
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),  # garante as 10 classes sempre
        target_names=class_names, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": report,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    normalize: bool = True,
) -> Path:
    """Desenha e grava a matriz de confusão como PNG.

    normalize=True mostra a fração por linha (recall por classe), mais legível
    quando as classes têm tamanhos diferentes.
    """
    labels = list(range(len(class_names)))
    cm = confusion_matrix(
        y_true, y_pred, labels=labels, normalize="true" if normalize else None
    )
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(ax=ax, cmap="Blues", values_format=".2f" if normalize else "d",
              xticks_rotation=45, colorbar=True)
    ax.set_title("Matriz de confusão" + (" (normalizada)" if normalize else ""))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def print_report(metrics: dict, class_names: list[str]) -> None:
    """Imprime um resumo legível na consola."""
    print(f"\nAccuracy : {metrics['accuracy']:.4f}")
    print(f"Macro-F1 : {metrics['macro_f1']:.4f}\n")
    print(f"{'classe':<22}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for name in class_names:
        row = metrics["per_class"][name]
        print(f"{name:<22}{row['precision']:>10.3f}{row['recall']:>10.3f}"
              f"{row['f1-score']:>10.3f}{int(row['support']):>10d}")
