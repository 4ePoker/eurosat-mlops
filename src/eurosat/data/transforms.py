"""Transforms (pré-processamento + data augmentation) para o EuroSAT.

Decisões pedagógicas importantes:

1. NORMALIZAÇÃO com estatísticas do ImageNet.
   Vamos usar backbones pré-treinados no ImageNet (via timm). Um modelo
   pré-treinado espera receber inputs distribuídos como os dados em que foi
   treinado. Por isso normalizamos com mean/std do ImageNet, não do EuroSAT.
   (Se treinássemos de raiz, usaríamos as estatísticas do próprio EuroSAT.)

2. RESIZE 64 -> 224.
   O EuroSAT RGB tem imagens 64x64. Os pesos pré-treinados do ImageNet foram
   aprendidos a 224x224; usar o mesmo tamanho aproveita melhor o backbone.

3. AUGMENTATION adequada a imagens de satélite.
   Imagens de satélite não têm orientação canónica ("cima/baixo") — uma
   floresta vista a norte é tão válida como vista a sul. Por isso flips
   horizontais/verticais e rotações de 90° são augmentations seguras e
   eficazes, ao contrário de, p.ex., fotos de pessoas onde virar ao contrário
   seria absurdo. Augmentation só se aplica ao TREINO, nunca a val/test.
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

# Estatísticas do ImageNet (RGB), convenção universal para modelos pré-treinados.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _base_transform(image_size: int) -> list:
    """Passos comuns a treino e avaliação: para tensor float + resize + normalize."""
    return [
        v2.ToImage(),  # PIL -> tv_tensor Image (API v2)
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float [0,1]
        v2.Resize((image_size, image_size), antialias=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def build_train_transform(image_size: int = 224) -> v2.Compose:
    """Transform de TREINO: base + augmentation estocástica."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((image_size, image_size), antialias=True),
            # Augmentation segura para imagens de satélite:
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=90),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224) -> v2.Compose:
    """Transform de VALIDAÇÃO/TESTE: determinístico, sem augmentation."""
    return v2.Compose(_base_transform(image_size))
