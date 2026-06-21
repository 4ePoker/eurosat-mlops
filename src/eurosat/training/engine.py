"""O "motor" do treino: uma época de treino e uma passagem de avaliação.

Separar o engine do entrypoint (`train.py`) é deliberado: aqui está a LÓGICA
(forward, loss, backward, step), testável e reutilizável; no train.py fica a
ORQUESTRAÇÃO (ler config, criar objetos, fazer logging). É o mesmo princípio de
desacoplamento da estrutura de pastas.

Escrevemos o loop à mão de propósito — é exatamente o que frameworks como o
Lightning escondem, e é o que querias ver para aprender.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from eurosat.training.metrics import AverageMeter, accuracy


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    limit_batches: int | None = None,
    on_step=None,
    log_every: int = 0,
) -> dict[str, float]:
    """Treina uma época. Devolve {'loss', 'acc'} médias da época.

    O ciclo canónico de treino em PyTorch, passo a passo:
      1. optimizer.zero_grad() — limpar gradientes do passo anterior
      2. forward -> logits
      3. criterion -> loss
      4. loss.backward() — calcular gradientes (autograd)
      5. optimizer.step() — atualizar os pesos
    O `scaler` (AMP) só faz diferença em GPU; em CPU é ignorado.

    Logging por step (curvas suaves): se `on_step` for dado e `log_every` > 0,
    chamamos `on_step(i, {"loss", "acc"})` a cada `log_every` batches. O engine
    fica agnóstico do W&B — quem faz o log é o callback, definido no train.py.
    """
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    use_amp = scaler is not None and device.type == "cuda"

    for i, (images, targets) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        batch_loss, batch_acc = loss.item(), accuracy(logits, targets)
        loss_meter.update(batch_loss, bs)
        acc_meter.update(batch_acc, bs)

        if on_step is not None and log_every and i % log_every == 0:
            on_step(i, {"loss": batch_loss, "acc": batch_acc})

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Avalia em val/teste (sem gradientes, modelo em modo eval)."""
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    for i, (images, targets) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        acc_meter.update(accuracy(logits, targets), bs)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}
