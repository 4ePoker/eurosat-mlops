"""Smoke-test do Passo 3(b): construção do modelo e forward pass."""

import torch

from eurosat.models import build_model, count_parameters
from eurosat.utils import seed_everything


def main() -> None:
    seed_everything(42)
    x = torch.randn(4, 3, 224, 224)  # batch falso, mesma forma do data module

    # 1) Fine-tuning (todos os pesos treináveis).
    model = build_model("resnet50", num_classes=10, pretrained=True)
    total, trainable = count_parameters(model)
    out = model(x)
    print(f"ResNet50 (fine-tuning): saída={tuple(out.shape)} "
          f"params={total:,} treináveis={trainable:,}")
    assert out.shape == (4, 10), "A cabeça deve produzir 10 logits"

    # 2) Feature extraction (backbone congelado).
    frozen = build_model("resnet50", num_classes=10, pretrained=True, freeze_backbone=True)
    _, frozen_trainable = count_parameters(frozen)
    print(f"ResNet50 (freeze):      treináveis={frozen_trainable:,} "
          f"({100 * frozen_trainable / total:.2f}% do total)")
    assert frozen_trainable < total, "Congelar deve reduzir os treináveis"
    assert frozen(x).shape == (4, 10)

    print("\n[OK] Smoke-test do modelo passou.")


if __name__ == "__main__":
    main()
