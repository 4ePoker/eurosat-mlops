"""Testes dos transforms.

O que garantimos:
* forma e dtype de saída corretos (o modelo espera (3, H, W) float32);
* a avaliação é DETERMINÍSTICA (sem augmentation) — crítico para métricas estáveis;
* o treino é ESTOCÁSTICO (a augmentation realmente muda a imagem);
* a normalização aplica de facto as estatísticas do ImageNet.
"""

import torch

from eurosat.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
)


def test_eval_transform_shape_and_dtype(random_pil_image):
    out = build_eval_transform(224)(random_pil_image())
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_eval_transform_is_deterministic(random_pil_image):
    img = random_pil_image()
    transform = build_eval_transform(224)
    assert torch.allclose(transform(img), transform(img))


def test_train_transform_is_stochastic(random_pil_image):
    # Com flips + rotação, duas aplicações à mesma imagem devem diferir.
    img = random_pil_image()
    transform = build_train_transform(224)
    torch.manual_seed(0)
    a = transform(img)
    b = transform(img)
    assert not torch.allclose(a, b)


def test_resize_respects_image_size(random_pil_image):
    out = build_eval_transform(128)(random_pil_image(size=64))
    assert out.shape == (3, 128, 128)


def test_normalization_uses_imagenet_stats(random_pil_image):
    # Imagem cinzenta constante (128) -> 128/255 antes de normalizar.
    img = random_pil_image(size=16, value=128)
    out = build_eval_transform(16)(img)
    scaled = 128 / 255.0
    for c in range(3):
        expected = (scaled - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert abs(out[c].mean().item() - expected) < 1e-3
