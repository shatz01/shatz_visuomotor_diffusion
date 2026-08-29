# Learning Diffusion for Manipulation

Small experiments for understanding diffusion policies, starting with toy diffusion models and gradually moving toward action generation for robot manipulation.

Create or update the shared project environment with:

```bash
uv sync
```

Check the local PyTorch/CUDA setup with:

```bash
uv run check_gpu.py
```

Inspect the Push-T environment and record a random-policy rollout:

```bash
uv run inspect_pusht.py
```

The rollout video is written to `outputs/pusht_inspection/`.

Download and inspect the official Push-T demonstrations:

```bash
mkdir -p data
curl -fL https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -o data/pusht.zip
echo "63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9  data/pusht.zip" | sha256sum --check
unzip -q data/pusht.zip -d data
uv run inspect_pusht_data.py
```

Dataset plots and an expert-episode video are written to `outputs/pusht_dataset/`.

Build the episode-safe `To=2`, `Tp=16`, `Ta=8` training and validation
dataloaders with:

```python
from pusht_dataset import create_pusht_dataloaders

loaders = create_pusht_dataloaders(batch_size=64)
batch = next(iter(loaders.train))
print(batch["observation"].shape)  # (64, 2, 5)
print(batch["action"].shape)       # (64, 16, 2)
```

Run the dataset tests with:

```bash
uv run pytest
```

Before full training, verify that the simple trajectory-regression MLP can
memorize one fixed batch:

```bash
uv run train_simple.py --overfit-one-batch
```

Then train it on the complete training split:

```bash
uv run train_simple.py
```

Each training run gets a unique directory named after its W&B run, such as
`outputs/simple_bc/jumping-fog-3-vm2b1036/`. Its best-validation checkpoint,
final checkpoint, and loss history are saved as `best.pt`, `last.pt`, and
`history.json` inside that directory.
Training and validation curves are also logged to the
`shatz-visuomotor-diffusion` project in Weights & Biases. To run without
uploading, pass `--wandb-mode disabled`; use `--wandb-mode offline` to save a
run for later synchronization. Training uses a fresh random seed per run while
retaining the same episode split; pass `--seed 42` to reproduce training exactly.
