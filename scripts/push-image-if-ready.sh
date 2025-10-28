#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-worker-0}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE="${REMOTE_USER:+$REMOTE_USER@}$SSH_HOST"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
FLAG="$SUBMIT_DIR/image_ready.flag"

LOCAL_IMAGE="${LOCAL_IMAGE:-gpu-cluster-acceptance:latest}"
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/dashabalashova/gpu-cluster-acceptance:latest}"

if [[ ! -f "$FLAG" ]]; then
  echo "Flag not found: $FLAG — nothing to do."
  exit 0
fi

echo "Flag found. Pushing $LOCAL_IMAGE -> $GHCR_IMAGE on $REMOTE"

ssh -o StrictHostKeyChecking=no "$REMOTE" "docker image inspect '$LOCAL_IMAGE' >/dev/null || { echo 'Image $LOCAL_IMAGE not found on remote'; exit 2; } \
 && docker tag '$LOCAL_IMAGE' '$GHCR_IMAGE' \
 && docker push '$GHCR_IMAGE'"
