"""Smoke-test do Passo 3(a): valida download, splits e shapes de um batch."""

from collections import Counter

from eurosat.data import EUROSAT_CLASSES, EuroSATDataModule
from eurosat.utils import seed_everything


def main() -> None:
    seed_everything(42)
    dm = EuroSATDataModule(batch_size=8, num_workers=0, image_size=224)

    print(">> A descarregar/preparar EuroSAT...")
    dm.prepare_data()
    dm.setup()

    n_train = len(dm.train_dataset)
    n_val = len(dm.val_dataset)
    n_test = len(dm.test_dataset)
    total = n_train + n_val + n_test
    print(f"\nSplits: train={n_train}  val={n_val}  test={n_test}  total={total}")
    assert total == 27000, "EuroSAT deve ter 27000 imagens"

    # Estratificação: distribuição de classes no treino deve ser ~uniforme.
    train_labels = [dm.train_dataset.dataset.samples[i][1] for i in dm.train_dataset.indices]
    dist = Counter(train_labels)
    print("\nDistribuição de classes no treino (estratificado):")
    for cls_idx in sorted(dist):
        print(f"  {EUROSAT_CLASSES[cls_idx]:<22} {dist[cls_idx]}")

    # Splits disjuntos (sem leakage).
    s_tr, s_va, s_te = set(dm.train_dataset.indices), set(dm.val_dataset.indices), set(dm.test_dataset.indices)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "Splits sobrepostos!"
    print("\n[OK] Splits disjuntos (sem data leakage).")

    # Shapes e normalização de um batch.
    images, labels = next(iter(dm.train_dataloader()))
    print(f"\nBatch de treino: images={tuple(images.shape)} dtype={images.dtype}  labels={tuple(labels.shape)}")
    print(f"Normalização -> média={images.mean():.3f} (esperado ~0)  std={images.std():.3f} (esperado ~1)")
    assert images.shape == (8, 3, 224, 224)
    assert labels.max() < 10 and labels.min() >= 0

    print("\n[OK] Smoke-test passou.")


if __name__ == "__main__":
    main()
