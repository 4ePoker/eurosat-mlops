"""Testes da lógica de dados: split estratificado e subset com transform.

São os testes mais importantes do projeto: é aqui que vivem os bugs silenciosos
de ML (leakage entre splits, desbalanceamento, splits não reprodutíveis).
Não descarregam o dataset — testam a lógica pura com dados sintéticos.
"""

from collections import Counter

import torch

from eurosat.data.datamodule import (
    EUROSAT_CLASSES,
    NUM_CLASSES,
    EuroSATDataModule,
    _stratified_split,
    _TransformedSubset,
)


def test_split_partitions_all_indices(balanced_targets):
    targets = balanced_targets(n_classes=4, per_class=25)  # 100 índices
    g = torch.Generator().manual_seed(1)
    train, val, test = _stratified_split(targets, 0.2, 0.2, g)
    # Cada índice aparece exatamente uma vez (cobertura + sem duplicados).
    assert sorted(train + val + test) == list(range(100))


def test_split_has_no_leakage(balanced_targets):
    targets = balanced_targets(n_classes=5, per_class=100)
    g = torch.Generator().manual_seed(42)
    train, val, test = _stratified_split(targets, 0.2, 0.2, g)
    s_tr, s_va, s_te = set(train), set(val), set(test)
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)


def test_split_is_stratified(balanced_targets):
    targets = balanced_targets(n_classes=5, per_class=100)
    g = torch.Generator().manual_seed(42)
    train, val, test = _stratified_split(targets, 0.2, 0.2, g)
    # Cada classe deve ter ~20 no val e ~20 no teste (20% de 100).
    val_counts = Counter(targets[i] for i in val)
    test_counts = Counter(targets[i] for i in test)
    for c in range(5):
        assert val_counts[c] == 20
        assert test_counts[c] == 20


def test_split_is_reproducible(balanced_targets):
    targets = balanced_targets(n_classes=3, per_class=50)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert _stratified_split(targets, 0.1, 0.1, g1) == _stratified_split(targets, 0.1, 0.1, g2)


def test_split_changes_with_seed(balanced_targets):
    targets = balanced_targets(n_classes=3, per_class=50)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    assert _stratified_split(targets, 0.1, 0.1, g1) != _stratified_split(targets, 0.1, 0.1, g2)


def test_split_is_shuffled_not_grouped_by_class(balanced_targets):
    # As classes não devem ficar agrupadas (o bug que o smoke-test apanhou).
    targets = balanced_targets(n_classes=5, per_class=100)
    g = torch.Generator().manual_seed(0)
    train, _, _ = _stratified_split(targets, 0.2, 0.2, g)
    first_labels = [targets[i] for i in train[:10]]
    assert len(set(first_labels)) > 1  # mistura de classes logo no início


class _FakeDataset:
    """Dataset mínimo: devolve (índice, índice*10) como (imagem, label)."""

    def __len__(self):
        return 100

    def __getitem__(self, i):
        return i, i * 10


def test_transformed_subset_length_and_indexing():
    subset = _TransformedSubset(_FakeDataset(), [3, 5, 7], transform=lambda x: x + 1000)
    assert len(subset) == 3
    image, label = subset[0]
    assert image == 1003  # transform aplicado
    assert label == 30  # label intacto


def test_transformed_subset_without_transform():
    subset = _TransformedSubset(_FakeDataset(), [1, 2], transform=None)
    image, label = subset[1]
    assert image == 2 and label == 20


def test_class_metadata_is_consistent():
    assert NUM_CLASSES == 10
    assert len(EUROSAT_CLASSES) == 10
    dm = EuroSATDataModule()
    assert dm.num_classes == 10
    assert dm.class_names == EUROSAT_CLASSES
