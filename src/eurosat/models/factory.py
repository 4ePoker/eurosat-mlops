"""Factory de modelos via timm.

Porquê uma *factory* (uma função que constrói o modelo) em vez de instanciar a
classe diretamente no código de treino? Para que a escolha de arquitetura seja
apenas mais um parâmetro de configuração. Trocar ResNet por ViT passa a ser
mudar uma string na config Hydra — o resto do pipeline não muda nada. Isto é o
desacoplamento "models não sabe de training" de que falámos.

Porquê timm (PyTorch Image Models)?
-----------------------------------
* Centenas de arquiteturas SOTA (ResNet, ViT, ConvNeXt, EfficientNet...) com
  pesos pré-treinados, atrás de UMA API consistente: `timm.create_model`.
* Substitui automaticamente a "cabeça" de classificação pelo número de classes
  que pedirmos (`num_classes=10`), tratando das dimensões internas por nós.
* Alternativas: `torchvision.models` (menos modelos, API menos uniforme) ou
  escrever a arquitetura à mão (educativo, mas reinventar a roda e sem pesos
  pré-treinados de qualidade).

TRANSFER LEARNING — as duas estratégias que esta factory suporta:
* Fine-tuning (default): treinar todos os pesos, partindo dos pré-treinados.
  Melhor accuracy quando há dados suficientes (o EuroSAT tem 27k, chega bem).
* Feature extraction (freeze_backbone=True): congelar o backbone e treinar só
  a cabeça nova. Muito mais rápido e bom para datasets pequenos; serve aqui
  como baseline barato para comparar.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(
    model_name: str = "resnet50",
    num_classes: int = 10,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Constrói um modelo de classificação pronto para o EuroSAT.

    Args:
        model_name: nome timm da arquitetura (ex.: "resnet50",
            "vit_base_patch16_224"). Ver `timm.list_models(pretrained=True)`.
        num_classes: nº de classes de saída (10 no EuroSAT).
        pretrained: carregar pesos pré-treinados no ImageNet.
        drop_rate: dropout antes da cabeça (regularização).
        freeze_backbone: se True, congela tudo menos a cabeça
            (estratégia de feature extraction).
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,  # timm troca a cabeça automaticamente
        drop_rate=drop_rate,
    )

    if freeze_backbone:
        # Congela todos os parâmetros...
        for param in model.parameters():
            param.requires_grad = False
        # ...e reativa apenas a cabeça de classificação.
        classifier = model.get_classifier()
        for param in classifier.parameters():
            param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Devolve (parâmetros totais, parâmetros treináveis).

    Útil para confirmar o efeito de `freeze_backbone` e para registar no W&B.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
