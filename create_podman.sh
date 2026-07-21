#!/usr/bin/env bash
# Build a Podman image for StreamDiffusionV2 with a working flash-attn install.
#
# This replicates, inside a container, the exact recipe that fixed flash-attn
# install failures on an RTX 4090 host:
#   1. Use a CUDA 12.4 "devel" base image (ships nvcc + headers + libs, so
#      CUDA_HOME is valid out of the box -- this is the thing that was missing
#      on the bare conda env and caused `OSError: CUDA_HOME environment
#      variable is not set`).
#   2. python 3.10 (project requires >=3.10,<3.13; host python 3.13 is too new).
#   3. `pip install streamdiffusionv2` -- this pulls torch==2.6.0, which on
#      PyPI resolves to the cu124 build. Do NOT install torch 2.11/torchvision
#      0.26 -- those are the Blackwell-only builds and this is an RTX 4090
#      (Ada), not Blackwell.
#   4. TMPDIR pointed at a dir on the same filesystem as the pip cache, to
#      dodge an `Invalid cross-device link` bug in flash-attn's setup.py when
#      it renames its downloaded wheel across filesystems (host issue was
#      /tmp on tmpfs vs ~/.cache/pip on ext4; kept here as a cheap safety net
#      in case the container runtime ever splits /tmp onto its own mount).
#   5. `pip install "streamdiffusionv2[flash-attn]" --no-build-isolation`,
#      which downloads a prebuilt flash-attn wheel matching
#      torch2.6+cu124+cp310 instead of compiling from source.
#
# Usage:
#   ./create_podman.sh              # build image
#   IMAGE_TAG=mytag ./create_podman.sh
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-streamdiffusionv2}"
IMAGE_TAG="${IMAGE_TAG:-cu124}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building ${IMAGE}"

if ! command -v podman >/dev/null 2>&1; then
  echo "podman not found. On Fedora: sudo dnf install -y podman" >&2
  exit 1
fi

# --- GPU access check (build doesn't need the GPU, but fail fast with a
# clear message so start_podman.sh doesn't surprise you later) -------------
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  cat >&2 <<'EOF'
WARNING: nvidia-container-toolkit (nvidia-ctk) not found.
start_podman.sh will not be able to pass the GPU into the container without it.

On Fedora, install it with:
  sudo dnf install -y nvidia-container-toolkit
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  nvidia-ctk cdi list   # sanity check, should list nvidia.com/gpu=all

Continuing with the image build; fix the above before running start_podman.sh.
EOF
elif [ ! -f /etc/cdi/nvidia.yaml ]; then
  echo "WARNING: /etc/cdi/nvidia.yaml missing. Generating it now (requires sudo)."
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
fi

CONTAINERFILE="${SCRIPT_DIR}/.streamdiffusionv2.Containerfile"

cat > "${CONTAINERFILE}" <<'EOF'
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH

# NOTE: deliberately no global ENV TMPDIR here. TMPDIR is read by every RUN
# command via mktemp -- including apt/dpkg postinst scripts (ca-certificates'
# postinst uses it) -- so setting it globally can break totally unrelated
# steps if the directory doesn't happen to exist yet at that point in the
# build. It's set locally, only on the one command that actually needs it,
# further down.

# ca-certificates' postinst (update-ca-certificates) is the flaky part in
# minimal container base images -- isolate it and recover from a failed
# trigger instead of letting it abort the whole layer.
RUN apt-get update -o Acquire::Retries=3 \
    && ( apt-get install -y --no-install-recommends ca-certificates \
         || (dpkg --configure -a && apt-get install -y -f) ) \
    && update-ca-certificates

# ffmpeg is intentionally NOT installed here: imageio-ffmpeg==0.6.0 (a pinned
# Python dependency of streamdiffusionv2) bundles its own static ffmpeg
# binary, so we don't need apt's ffmpeg + its libavdevice/libavfilter/etc.
# dependency chain at all.
RUN apt-get install -y --no-install-recommends wget git \
    && rm -rf /var/lib/apt/lists/*

# Miniconda, so we get an isolated python 3.10 matching the project's
# requires-python (>=3.10,<3.13) regardless of what the base image ships.
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Anaconda now requires explicit ToS acceptance for its default channels
# before conda will operate on them non-interactively.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n streamdiffusionv2 python=3.10 -y

SHELL ["conda", "run", "--no-capture-output", "-n", "streamdiffusionv2", "/bin/bash", "-c"]

# streamdiffusionv2 pulls torch==2.6.0, which resolves to the cu124 wheel on
# PyPI for linux -- correct for Ada (RTX 4090), do not add the Blackwell
# torch==2.11.0 / torchvision==0.26.0 pin here.
RUN pip install --no-cache-dir streamdiffusionv2 ninja packaging

# nvcc from the base image + CUDA_HOME above lets flash-attn's setup.py
# generate build metadata; it then auto-downloads a prebuilt wheel matching
# torch2.6+cu124+cp310 instead of compiling from source. TMPDIR is scoped to
# just this command (see note above ENV) and pointed at a dir on the same
# filesystem as pip's wheel cache (~/.cache/pip), to dodge an
# `Invalid cross-device link` bug in flash-attn's setup.py if the build's
# temp dir and the wheel cache ever end up on different mounts.
RUN mkdir -p /root/.cache/pip-tmp \
    && TMPDIR=/root/.cache/pip-tmp pip install --no-cache-dir "streamdiffusionv2[flash-attn]" --no-build-isolation

# Drop back to a normal shell for CMD/entrypoint purposes.
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate streamdiffusionv2" >> /root/.bashrc

WORKDIR /workspace
EXPOSE 7860
CMD ["/bin/bash"]
EOF

podman build -t "${IMAGE}" -f "${CONTAINERFILE}" "${SCRIPT_DIR}"

echo "==> Verifying flash-attn inside the built image"
podman run --rm "${IMAGE}" conda run -n streamdiffusionv2 python -c "
import torch, flash_attn
print('torch:', torch.__version__)
print('flash_attn:', flash_attn.__version__)
"

echo "==> Built ${IMAGE}. Run ./start_podman.sh to launch it with GPU access."
