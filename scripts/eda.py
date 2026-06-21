"""Análise Exploratória de Dados (EDA) do EuroSAT.

Responde, sobre os dados CRUS, a:
  1. Que classes existem e o que significam?
  2. Quantas imagens por classe? Está balanceado?
  3. Qual o formato das imagens (tamanho, canais, dtype, gama de valores)?
  4. Como são as imagens de cada classe? (grelha de amostras)
  5. Como se distribuem as intensidades de pixel por canal RGB?

Gera figuras em assets/eda/ e imprime um resumo no terminal.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import EuroSAT

OUT = Path("assets/eda")
OUT.mkdir(parents=True, exist_ok=True)

# Significado de cada classe (uso/cobertura do solo — land use / land cover).
CLASS_DESCRIPTIONS = {
    "AnnualCrop": "Culturas anuais (ex.: cereais, sazonais)",
    "Forest": "Floresta / árvores densas",
    "HerbaceousVegetation": "Vegetação herbácea (arbustos, ervas)",
    "Highway": "Autoestradas / grandes vias",
    "Industrial": "Zonas industriais",
    "Pasture": "Pastagens",
    "PermanentCrop": "Culturas permanentes (ex.: pomares, vinhas)",
    "Residential": "Zonas residenciais",
    "River": "Rios",
    "SeaLake": "Mar / lagos",
}


def main() -> None:
    ds = EuroSAT(root="data/raw", download=False)
    classes = ds.classes
    labels = [label for _, label in ds.samples]
    counts = Counter(labels)

    # --- 1 & 2: classes e distribuição ---
    print("=" * 64)
    print(f"EuroSAT — {len(ds)} imagens, {len(classes)} classes")
    print("=" * 64)
    print(f"{'classe':<22}{'nº imagens':>12}{'%':>8}   descrição")
    for i, name in enumerate(classes):
        n = counts[i]
        print(f"{name:<22}{n:>12}{100*n/len(ds):>7.1f}%   {CLASS_DESCRIPTIONS[name]}")
    imbalance = max(counts.values()) / min(counts.values())
    print(f"\nRácio de desequilíbrio (maior/menor classe): {imbalance:.2f}x")

    # --- 3: formato das imagens ---
    img0, _ = ds[0]
    arr0 = np.asarray(img0)
    print(f"\nFormato da imagem: {arr0.shape} (altura, largura, canais)")
    print(f"dtype: {arr0.dtype}   gama de valores: [{arr0.min()}, {arr0.max()}]")

    # --- Figura A: distribuição de classes ---
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(classes)
    values = [counts[i] for i in range(len(classes))]
    ax.bar(names, values, color="steelblue")
    ax.set_title("Distribuição de imagens por classe — EuroSAT")
    ax.set_ylabel("nº de imagens")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v + 30, str(v), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "class_distribution.png", dpi=120)
    plt.close(fig)

    # --- Figura B: grelha de amostras (1 linha por classe) ---
    rng = np.random.default_rng(42)
    by_class: dict[int, list[int]] = {i: [] for i in range(len(classes))}
    for idx, label in enumerate(ds.samples):
        by_class[label[1]].append(idx)
    n_per = 5
    fig, axes = plt.subplots(len(classes), n_per, figsize=(n_per * 1.6, len(classes) * 1.6))
    for r, name in enumerate(classes):
        picks = rng.choice(by_class[r], size=n_per, replace=False)
        for c, idx in enumerate(picks):
            img, _ = ds[idx]
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.suptitle("Amostras por classe — EuroSAT (64×64 RGB)")
    fig.tight_layout()
    fig.savefig(OUT / "samples_grid.png", dpi=120)
    plt.close(fig)

    # --- Figura C: histograma de intensidades por canal RGB ---
    sample_idx = rng.choice(len(ds), size=min(2000, len(ds)), replace=False)
    pixels = np.stack([np.asarray(ds[i][0]) for i in sample_idx])  # (N,64,64,3)
    fig, ax = plt.subplots(figsize=(9, 5))
    for ch, color in zip(range(3), ["red", "green", "blue"]):
        ax.hist(pixels[..., ch].ravel(), bins=64, range=(0, 255),
                color=color, alpha=0.5, label=color, density=True)
    ax.set_title("Distribuição de intensidade de pixel por canal (amostra de 2000 imgs)")
    ax.set_xlabel("valor do pixel (0–255)")
    ax.set_ylabel("densidade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pixel_histograms.png", dpi=120)
    plt.close(fig)

    # Médias por canal (úteis para perceber a normalização).
    print("\nMédia/desvio por canal (na amostra, escala 0–1):")
    for ch, color in zip(range(3), ["R", "G", "B"]):
        vals = pixels[..., ch] / 255.0
        print(f"  {color}: média={vals.mean():.3f}  std={vals.std():.3f}")

    print(f"\nFiguras guardadas em {OUT}/")


if __name__ == "__main__":
    main()
