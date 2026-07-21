#!/usr/bin/env bash
# Build a minimal Podman image with just CUDA + the correct PyTorch build.
# No conda: the container is already the isolation boundary. Everything else
# (streamdiffusionv2, flash-attn, etc.) is left for you to add on top.
#
# - Base: nvidia/cuda:12.4.1-devel-ubuntu22.04 -- ships nvcc + CUDA 12.4
#   headers/libs, matching the torch build below. Ubuntu 22.04's system
#   python3 is 3.10, which satisfies this project's requires-python
#   (>=3.10,<3.13) with no conda needed.
# - torch==2.6.0 / torchvision==0.21.0 / torchaudio==2.6.0 pinned exactly as
#   in this repo's pyproject.toml, pulled explicitly from the cu124 wheel
#   index so the CUDA build is guaranteed (not left to pip's default
#   resolution). Ada (RTX 4090) -- do not swap in the Blackwell-only
#   torch==2.11.0 / torchvision==0.26.0 build.
#
# Usage:
#   ./create_podman.sh
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-streamdiffusionv2}"
IMAGE_TAG="${IMAGE_TAG:-cu124}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v podman >/dev/null 2>&1; then
  echo "podman not found. On Fedora: sudo dnf install -y podman" >&2
  exit 1
fi

CONTAINERFILE="${SCRIPT_DIR}/.streamdiffusionv2.Containerfile"

cat > "${CONTAINERFILE}" <<'EOF'
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

# ca-certificates' postinst (update-ca-certificates) is the flaky part in
# minimal container base images -- isolate it and recover from a failed
# trigger instead of letting it abort the whole layer.
RUN apt-get update -o Acquire::Retries=3 \
    && ( apt-get install -y --no-install-recommends ca-certificates \
         || (dpkg --configure -a && apt-get install -y -f) ) \
    && update-ca-certificates

RUN apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

WORKDIR /workspace
CMD ["/bin/bash"]
EOF

podman build -t "${IMAGE}" -f "${CONTAINERFILE}" "${SCRIPT_DIR}"

echo "==> Verifying torch inside the built image (no GPU during build, so cuda.is_available() is expected False here)"
podman run --rm "${IMAGE}" python -c "
import torch
print('torch:', torch.__version__)
print('torch.version.cuda:', torch.version.cuda)
"

echo "==> Built ${IMAGE}."
