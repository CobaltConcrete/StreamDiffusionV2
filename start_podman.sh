#!/usr/bin/env bash
# Start the StreamDiffusionV2 podman container built by create_podman.sh,
# with GPU access and the repo / checkpoints bind-mounted in.
#
# Usage:
#   ./start_podman.sh                 # interactive shell (conda env pre-activated)
#   ./start_podman.sh ./run_v2v.sh single   # run a command directly
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-streamdiffusionv2}"
IMAGE_TAG="${IMAGE_TAG:-cu124}"
IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="${CONTAINER_NAME:-streamdiffusionv2}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v podman >/dev/null 2>&1; then
  echo "podman not found. On Fedora: sudo dnf install -y podman" >&2
  exit 1
fi

if ! podman image exists "${IMAGE}"; then
  echo "Image ${IMAGE} not found. Run ./create_podman.sh first." >&2
  exit 1
fi

GPU_ARGS=()
if command -v nvidia-ctk >/dev/null 2>&1 && [ -f /etc/cdi/nvidia.yaml ]; then
  GPU_ARGS=(--device nvidia.com/gpu=all --security-opt=label=disable)
else
  echo "WARNING: no CDI nvidia spec found (/etc/cdi/nvidia.yaml)." >&2
  echo "GPU will NOT be available inside the container. See create_podman.sh for setup steps." >&2
fi

# Checkpoints/models can be large -- keep them on the host, outside the repo
# checkout, so `podman rm`/rebuilds never touch downloaded weights.
DATA_DIR="${DATA_DIR:-$HOME/.cache/streamdiffusionv2}"
mkdir -p "${DATA_DIR}/ckpts" "${DATA_DIR}/wan_models" "${DATA_DIR}/outputs"

if podman container exists "${CONTAINER_NAME}"; then
  echo "==> Removing existing container ${CONTAINER_NAME}"
  podman rm -f "${CONTAINER_NAME}" >/dev/null
fi

exec podman run -it --rm \
  --name "${CONTAINER_NAME}" \
  "${GPU_ARGS[@]}" \
  --userns=keep-id \
  --shm-size=8g \
  -p 7860:7860 \
  -v "${SCRIPT_DIR}:/workspace:Z" \
  -v "${DATA_DIR}/ckpts:/workspace/ckpts:Z" \
  -v "${DATA_DIR}/wan_models:/workspace/wan_models:Z" \
  -v "${DATA_DIR}/outputs:/workspace/outputs:Z" \
  -w /workspace \
  "${IMAGE}" \
  conda run --no-capture-output -n streamdiffusionv2 "${@:-bash}"
