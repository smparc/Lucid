# ─────────────────────────────────────────────────────────────────────────────
# Lucid: Accelerated MRI Reconstruction
# Multi-stage build for minimal image size
# ─────────────────────────────────────────────────────────────────────────────


FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*


RUN ln -sf /usr/bin/python3.10 /usr/bin/python


WORKDIR /app


# ── Dependencies ─────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Application code ─────────────────────────────────────────────────────────
COPY . .


# ── Volumes for data and outputs ─────────────────────────────────────────────
VOLUME ["/app/data", "/app/outputs"]


# ── Default: show help ────────────────────────────────────────────────────────
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]