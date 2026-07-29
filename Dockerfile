# =============================================================================
# Lucid: accelerated MRI reconstruction
#
# Multi-stage build. The builder installs into a virtualenv which is copied into
# a clean runtime image, so build tooling and pip caches never reach the final
# layer.
#
# Build:
#   docker build -t lucid-mri .
#   docker build -t lucid-mri-cpu --build-arg BASE=ubuntu:22.04 .
#
# Run:
#   docker run --gpus all -v ./data:/app/data -v ./outputs:/app/outputs \
#       lucid-mri train --config configs/swinunet_dc.yaml
# =============================================================================

ARG BASE=nvidia/cuda:12.1.0-runtime-ubuntu22.04

# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------
FROM ${BASE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies are installed before the source is copied, so editing code does
# not invalidate the (slow) dependency layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --------------------------------------------------------------------------
# Runtime
# --------------------------------------------------------------------------
FROM ${BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        libhdf5-103 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Run as a non-root user. Training writes checkpoints into mounted volumes, and
# a root-owned outputs/ directory is a persistent nuisance on the host.
RUN useradd --create-home --uid 1000 lucid \
    && mkdir -p /app/data /app/outputs \
    && chown -R lucid:lucid /app

COPY --chown=lucid:lucid . .

USER lucid

VOLUME ["/app/data", "/app/outputs"]

# Fails the build if the image cannot construct every architecture.
RUN python main.py test_models --size 64

HEALTHCHECK --interval=60s --timeout=20s --start-period=10s --retries=2 \
    CMD python -c "import torch, models, data, training; print('ok')" || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
