"""Entrypoint de avaliação. Carrega um checkpoint e avalia no TESTE.

Correr:
    PYTHONPATH=src python -m eurosat.evaluation.run \
        checkpoint=outputs/2026-06-21/13-30-41/best_model.pt

Reutiliza a MESMA config (data/model) por composição do Hydra, garantindo que
o modelo é reconstruído exatamente como no treino antes de carregar os pesos.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from eurosat.evaluation.evaluate import (
    collect_predictions,
    compute_metrics,
    plot_confusion_matrix,
)
from eurosat.evaluation.evaluate import print_report
from eurosat.utils import seed_everything


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    if cfg.checkpoint is None:
        raise ValueError("Indica o checkpoint: checkpoint=caminho/para/best_model.pt")

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(HydraConfig.get().runtime.output_dir)

    # Dados (só precisamos do teste) e modelo, reconstruídos a partir da config.
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    class_names = list(datamodule.class_names)

    model = hydra.utils.instantiate(cfg.model).to(device)
    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Checkpoint carregado: {cfg.checkpoint} (val_acc={ckpt.get('val_acc')})")

    # Previsões + métricas + matriz de confusão.
    y_true, y_pred = collect_predictions(
        model, test_loader, device, limit_batches=cfg.limit_batches
    )
    metrics = compute_metrics(y_true, y_pred, class_names)
    print_report(metrics, class_names)

    cm_path = plot_confusion_matrix(y_true, y_pred, class_names, out_dir / "confusion_matrix.png")
    print(f"\nMatriz de confusão guardada em {cm_path}")

    # Logging W&B (respeita o mode da config).
    wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode,
        name=(cfg.wandb.name or "eval"), tags=list(cfg.wandb.tags) + ["eval"],
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb.log({
        "test/accuracy": metrics["accuracy"],
        "test/macro_f1": metrics["macro_f1"],
        "test/confusion_matrix": wandb.Image(str(cm_path)),
    })
    wandb.finish()

    return metrics["accuracy"]


if __name__ == "__main__":
    main()
