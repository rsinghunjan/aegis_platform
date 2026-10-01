
# Lightweight tools image used by k8s jobs (backup/restore, loadtest) and local bench
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps commonly used by example scripts
RUN pip install --no-cache-dir numpy torch torchvision onnx onnxruntime requests

# Install cosign (for signing in CI or local dev)
RUN curl -sSLf https://github.com/sigstore/cosign/releases/download/v1.14.1/cosign-linux-amd64 -o /usr/local/bin/cosign \
    && chmod +x /usr/local/bin/cosign

WORKDIR /opt/aegis
COPY . /opt/aegis

ENTRYPOINT ["/bin/sh", "-c"]
