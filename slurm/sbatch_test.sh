#!/bin/bash

#SBATCH --job-name=train-ddp
#SBATCH --nodes=4
#SBATCH -D .
#SBATCH --output=logs/O-%x_%j.txt
#SBATCH --error=logs/E-%x_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL


mkdir -p logs
cd $SLURM_SUBMIT_DIR

export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NNODES
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

start_ts="$(date '+%Y-%m-%d %H:%M:%S')"
SECONDS=0
finish() {
  local elapsed=$SECONDS
  printf "Job started: %s\nJob ended:   %s\nElapsed:     %02d:%02d:%02d\n" \
    "$start_ts" "$(date '+%Y-%m-%d %H:%M:%S')" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
}
trap finish EXIT

_job_rc=0
finish() {
  local rc=${_job_rc:-$?}
  local elapsed=$SECONDS
  printf "\nJob started: %s\nJob ended:   %s\nElapsed:     %02d:%02d:%02d\n" \
    "$start_ts" "$(date '+%Y-%m-%d %H:%M:%S')" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
  printf "Job exit code: %s\n" "$rc"
  if [ "$rc" -eq 0 ]; then
    echo "✅ JOB SUCCESS"
  else
    echo "❌ JOB FAILED"
  fi
}
trap finish EXIT

# srun --container-image="$HOME/gpu-cluster-acceptance/docker/gpu-cluster-acceptance.sqsh" \
#     -- bash -lc 'python3 /workspace/train.py'

srun --container-image="$HOME/gpu-cluster-acceptance/docker/gpu-cluster-acceptance.sqsh" \
    --container-mounts="$SLURM_SUBMIT_DIR/docker:/workspace" \
    -- bash -lc 'python3 /workspace/train.py'

_job_rc=$?
if [ "$_job_rc" -eq 0 ]; then
  echo "PASSED" > "$SLURM_SUBMIT_DIR/test_result.txt"
  touch "$SLURM_SUBMIT_DIR/image_ready.flag"
else
  echo "FAILED" > "$SLURM_SUBMIT_DIR/test_result.txt"
  exit $_job_rc
fi
