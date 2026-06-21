"""Entrypoint de treino. Orquestra config (Hydra) + dados + modelo + W&B.

Correr:
    PYTHONPATH=src python -m eurosat.training.train
    PYTHONPATH=src python -m eurosat.training.train model=vit training.epochs=10
    PYTHONPATH=src python -m eurosat.training.train wandb.mode=online

O `@hydra.main` injeta o `cfg` já composto a partir de conf/config.yaml e cria
um diretório de saída por execução, onde guarda a config exata usada.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from eurosat.models import count_parameters
from eurosat.training.engine import evaluate, train_one_epoch
from eurosat.utils import seed_everything


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    # Só passamos parâmetros treináveis: respeita freeze_backbone automaticamente.
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.training.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.training.lr,
                                 weight_decay=cfg.training.weight_decay)
    if cfg.training.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.training.lr, momentum=0.9,
                               weight_decay=cfg.training.weight_decay)
    raise ValueError(f"optimizer desconhecido: {cfg.training.optimizer}")


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    device = _resolve_device(cfg.training.device)
    print(f"Device: {device}")

    # --- W&B: um único ponto de inicialização; o modo vem da config. ---
    # Passamos a config inteira para o W&B: assim cada run fica ligada aos
    # hiperparâmetros exatos que a geraram (reprodutibilidade de experiências).
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    # Eixo-x comum: número de batches vistos. Assim as métricas por-step (treino)
    # e por-época (val) alinham-se no mesmo eixo, dando curvas suaves no W&B.
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    wandb.define_metric("epoch", step_metric="global_step")

    # --- Dados e modelo: instanciados a partir das configs via _target_. ---
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = hydra.utils.instantiate(cfg.model).to(device)
    total, trainable = count_parameters(model)
    print(f"Modelo: {total:,} params ({trainable:,} treináveis)")

    optimizer = _build_optimizer(cfg, model)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    scheduler = None
    if cfg.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp and device.type == "cuda")

    # --- O loop de treino. ---
    best_acc = 0.0
    # Diretório desta run (robusto, independentemente de o Hydra mudar o cwd).
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    steps_per_epoch = len(train_loader)
    for epoch in range(1, cfg.training.epochs + 1):
        # Callback de logging por step: regista a loss/acc do batch no W&B,
        # com global_step = batches já vistos no total (eixo-x contínuo).
        def on_step(i: int, m: dict, _epoch: int = epoch) -> None:
            gs = (_epoch - 1) * steps_per_epoch + i
            wandb.log({"train/loss_step": m["loss"], "train/acc_step": m["acc"],
                       "global_step": gs})

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            limit_batches=cfg.training.limit_train_batches,
            on_step=on_step, log_every=cfg.training.log_every_n_steps,
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            limit_batches=cfg.training.limit_val_batches,
        )
        if scheduler is not None:
            scheduler.step()

        # Logging: consola + W&B na mesma chamada lógica.
        log = {
            "epoch": epoch,
            "global_step": epoch * steps_per_epoch,
            "train/loss": train_metrics["loss"],
            "train/acc": train_metrics["acc"],
            "val/loss": val_metrics["loss"],
            "val/acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        wandb.log(log)
        print(f"[{epoch:02d}/{cfg.training.epochs}] "
              f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}")

        # Checkpoint do melhor modelo (por val acc).
        if cfg.training.save_checkpoint and val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch,
                 "val_acc": best_acc, "config": OmegaConf.to_container(cfg, resolve=True)},
                out_dir / "best_model.pt",
            )

    wandb.summary["best_val_acc"] = best_acc
    wandb.finish()
    print(f"\nMelhor val acc: {best_acc:.4f}  (checkpoint em {out_dir})")
    return best_acc


if __name__ == "__main__":
    main()
