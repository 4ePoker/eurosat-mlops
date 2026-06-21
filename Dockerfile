# syntax=docker/dockerfile:1
#
# Imagem reprodutível do pipeline EuroSAT.
#
# Decisões:
# * Base `python:3.11-slim` — leve e com a versão de Python que fixámos.
# * Instalamos PRIMEIRO o torch/torchvision da CPU (índice dedicado) para a
#   imagem não trazer ~2GB de CUDA que não usaríamos numa máquina sem GPU.
#   Como já satisfazem os requisitos do pyproject, o `pip install .` seguinte
#   não os reinstala.
# * Ordem das camadas pensada para CACHE: dependências mudam raramente (camada
#   reutilizada), código muda muito (camada no fim).
# * A WANDB_API_KEY NUNCA é gravada na imagem — passa-se em runtime via
#   `docker run -e WANDB_API_KEY=...`. (Segredos em imagens = fuga garantida.)

FROM python:3.11-slim

# libgomp1: runtime de OpenMP exigido pelo PyTorch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# 1) Torch/torchvision CPU numa camada própria (a mais pesada e estável).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

# 2) Restantes dependências, instaladas a partir do pyproject.
#    Copiamos só os metadados + código mínimo necessário para o build.
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[dev]"

# 3) Configs e testes (mudam mais, ficam no fim para aproveitar o cache acima).
COPY conf ./conf
COPY tests ./tests

# Os dados (versionados por DVC) montam-se em runtime, não vão na imagem.
VOLUME ["/app/data"]

# Corremos a partir do código-fonte (PYTHONPATH=/app/src) para que o Hydra
# encontre /app/conf pelo caminho relativo dos entrypoints.
#
# Exemplos de uso:
#   Treino:     docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY \
#                   -v $PWD/data:/app/data eurosat-mlops
#   ViT/online: docker run --rm -e WANDB_API_KEY=$WANDB_API_KEY \
#                   -v $PWD/data:/app/data eurosat-mlops model=vit
#   Testes:     docker run --rm --entrypoint pytest eurosat-mlops
ENTRYPOINT ["python", "-m", "eurosat.training.train"]
CMD []
