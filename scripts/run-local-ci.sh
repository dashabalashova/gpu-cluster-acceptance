#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

WORKDIR="$HOME/gpu-cluster-acceptance"
DOCKER_DIR="docker"
SSH_HOST="${SSH_HOST:-worker-0}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE="${REMOTE_USER:+$REMOTE_USER@}$SSH_HOST"
LOCAL_IMAGE="${LOCAL_IMAGE:-gpu-cluster-acceptance:latest}"
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/dashabalashova/gpu-cluster-acceptance:latest}"

TIMEOUT_SECS="${TIMEOUT_SECS:-180}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

# --- helper ---
log(){ printf '\n[run_local_ci] %s\n' "$*"; }

# 1) Build image on worker (ssh -> cd -> docker build -> optional enroot import)
log "SSH to $REMOTE: build docker image from $WORKDIR/$DOCKER_DIR"
ssh -o StrictHostKeyChecking=no "$REMOTE" bash -lc "'
set -euo pipefail
cd \"$WORKDIR/$DOCKER_DIR\"
echo \"[remote] Building docker image: $LOCAL_IMAGE\"
docker build -t \"$LOCAL_IMAGE\" .
if command -v enroot >/dev/null 2>&1; then
  echo \"[remote] enroot found -> creating .sqsh\"
  enroot import --output \"gpu-cluster-acceptance.sqsh\" \"dockerd://$LOCAL_IMAGE\" || echo \"[remote] enroot import failed (non-fatal)\"
else
  echo \"[remote] enroot not found\"
fi
'"

log "Remote build finished."

# 2) Back on login: submit sbatch
log "Submitting sbatch ($WORKDIR/slurm/sbatch_test.sh)"
cd "$WORKDIR"
sbatch slurm/sbatch_test.sh
log "sbatch submitted. Now polling for test result (image_ready.flag or test_result.txt) up to $TIMEOUT_SECS s"

# 3) Poll for result file (shared FS assumed)
end=$(( $(date +%s) + TIMEOUT_SECS ))
while true; do
  now=$(date +%s)
  if [[ $now -ge $end ]]; then
    log "TIMEOUT waiting for image_ready.flag (waited $TIMEOUT_SECS s). Exiting with failure."
    exit 2
  fi

  if [[ -f "$WORKDIR/image_ready.flag" ]]; then
    log "Found image_ready.flag — test passed on cluster."
    break
  fi

  if [[ -f "$WORKDIR/test_result.txt" ]]; then
    val=$(<"$WORKDIR/test_result.txt")
    log "Found test_result.txt -> $val"
    if [[ "$val" == "PASSED" ]]; then
      [[ -f "$WORKDIR/image_ready.flag" ]] || touch "$WORKDIR/image_ready.flag"
      break
    else
      log "Test result indicates failure: $val"
      exit 3
    fi
  fi

  sleep "$POLL_INTERVAL"
done

# 4) Show last logs (optional)
if ls "$WORKDIR/logs"/*-train-ddp_*.txt >/dev/null 2>&1; then
  log "Тail 200:"
  tail -n 200 "$WORKDIR/logs"/*-train-ddp_*.txt || true
else
  log "No logs in $WORKDIR/logs."
fi

# 5) Login to ghcr and push
if [[ -z "${GHCR_PAT:-}" ]]; then
  log '  export GHCR_PAT="ghp_..."'
else
  log "Logging in to ghcr.io"
  echo "$GHCR_PAT" | docker login ghcr.io -u "${GITHUB_USER:-$USER}" --password-stdin
fi

# 6) Call push_if_ready.sh
log "Calling push_if_ready.sh to tag & push image from remote ($SSH_HOST)"
chmod +x scripts/push-image-if-ready.sh
SSH_HOST="$SSH_HOST" REMOTE_USER="$REMOTE_USER" LOCAL_IMAGE="$LOCAL_IMAGE" GHCR_IMAGE="$GHCR_IMAGE" scripts/push-image-if-ready.sh

log "Done."
