"""DataModule do EuroSAT: download, split reprodutível e dataloaders.

Padrão "DataModule": juntar num único objeto toda a lógica de dados —
onde estão, como se dividem em treino/val/teste, e como se servem em batches.
Isto mantém o resto do código (modelo, treino) ignorante dos detalhes dos dados,
que era o desacoplamento que discutimos na estrutura de pastas.

Pontos de MLOps a notar:

* SPLIT ESTRATIFICADO E SEMEADO. O EuroSAT não traz splits oficiais. Criamos
  treino/val/teste nós próprios, de forma *estratificada* (mantendo a proporção
  de cada classe em cada split) e *semeada* (a mesma seed -> exatamente o mesmo
  split). Sem isto, a comparação entre experiências é inválida: estaríamos a
  comparar modelos avaliados em conjuntos de teste diferentes.

* TRANSFORMS DIFERENTES POR SPLIT. Treino leva augmentation; val/teste não.
  Como o dataset base é um só, usamos um wrapper (`_TransformedSubset`) que
  aplica o transform certo a cada subconjunto.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import EuroSAT

from eurosat.data.transforms import build_eval_transform, build_train_transform

# As 10 classes do EuroSAT (ordem alfabética, igual à do torchvision).
EUROSAT_CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
NUM_CLASSES = len(EUROSAT_CLASSES)


class _TransformedSubset(Dataset):
    """Subset que aplica o seu próprio transform.

    Necessário porque queremos augmentation só no treino, partilhando o mesmo
    dataset base entre os três splits.
    """

    def __init__(self, dataset: Dataset, indices: list[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _stratified_split(
    targets: list[int],
    val_fraction: float,
    test_fraction: float,
    generator: torch.Generator,
) -> tuple[list[int], list[int], list[int]]:
    """Divide índices em treino/val/teste mantendo a proporção por classe."""
    rng = np.random.default_rng(generator.initial_seed())
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(by_class):
        idxs = np.array(by_class[label])
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test : n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val :].tolist())

    # Baralhar cada split (determinístico) para não ficarem agrupados por classe:
    # importante para que uma avaliação parcial (limit_batches) seja representativa.
    for split in (train_idx, val_idx, test_idx):
        rng.shuffle(split)
    return train_idx, val_idx, test_idx


@dataclass
class EuroSATDataModule:
    """Configura e serve os dados do EuroSAT.

    Args:
        data_dir: onde guardar/procurar o dataset.
        image_size: tamanho final das imagens (224 para backbones ImageNet).
        batch_size: tamanho do batch.
        num_workers: processos de carregamento paralelo.
        val_fraction / test_fraction: frações para validação e teste.
        seed: semente do split (reprodutibilidade).
        download: se deve descarregar o dataset caso não exista.
    """

    data_dir: str = "data/raw"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    download: bool = True

    # preenchidos em setup()
    train_dataset: Dataset | None = None
    val_dataset: Dataset | None = None
    test_dataset: Dataset | None = None

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def class_names(self) -> tuple[str, ...]:
        return EUROSAT_CLASSES

    def prepare_data(self) -> None:
        """Só descarrega (operação a fazer uma vez, sem estado partilhado)."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        if self.download:
            EuroSAT(root=self.data_dir, download=True)

    def setup(self) -> None:
        """Cria os três splits com os respetivos transforms."""
        base = EuroSAT(root=self.data_dir, download=False)
        targets = [label for _, label in base.samples]  # rápido: lê só os labels

        generator = torch.Generator().manual_seed(self.seed)
        train_idx, val_idx, test_idx = _stratified_split(
            targets, self.val_fraction, self.test_fraction, generator
        )

        train_tf = build_train_transform(self.image_size)
        eval_tf = build_eval_transform(self.image_size)
        self.train_dataset = _TransformedSubset(base, train_idx, train_tf)
        self.val_dataset = _TransformedSubset(base, val_idx, eval_tf)
        self.test_dataset = _TransformedSubset(base, test_idx, eval_tf)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,  # só no treino, para batches consistentes
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)
