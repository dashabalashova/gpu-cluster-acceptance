# gpu-cluster-acceptance


```
export GHCR_PAT="ghp_..."

git clone https://github.com/dashabalashova/gpu-cluster-acceptance.git
cd gpu-cluster-acceptance
```

```
chmod +x scripts/run-local-ci.sh
scripts/run-local-ci.sh
```

```
WORLD_SIZE=4
MASTER_ADDR=worker-3
[2025-10-28T02:23:53.551905] initializing process group: backend=nccl, rank=2, world_size=4, local_rank=0, cuda=True
[2025-10-28T02:23:56.317367] initializing process group: backend=nccl, rank=3, world_size=4, local_rank=0, cuda=True
[2025-10-28T02:23:56.343494] initializing process group: backend=nccl, rank=0, world_size=4, local_rank=0, cuda=True
[2025-10-28T02:23:57.902124] initializing process group: backend=nccl, rank=1, world_size=4, local_rank=0, cuda=True
Epoch 1/3 - train_loss=0.6884 train_acc=0.5287
Epoch 1/3 - val_loss=0.6835 val_acc=0.5340
Epoch 2/3 - train_loss=0.6834 train_acc=0.5549
Epoch 2/3 - val_loss=0.6770 val_acc=0.5558
Epoch 3/3 - train_loss=0.6721 train_acc=0.5994
Epoch 3/3 - val_loss=0.6722 val_acc=0.5825
Training complete. elapsed=8.1s

Job started: 2025-10-28 02:23:33
Job ended:   2025-10-28 02:24:07
Elapsed:     00:00:34
Job exit code: 0
✅ JOB SUCCESS
```