# Learning Diffusion for Manipulation

Small experiments for understanding diffusion policies, starting with toy diffusion models and gradually moving toward action generation for robot manipulation.

Create or update the shared project environment with:

```bash
uv sync
```

Check the local PyTorch/CUDA setup with:

```bash
uv run python -m src.compute.check_gpu
```

Inspect the Push-T environment and record a random-policy rollout:

```bash
uv run python -m src.tools.inspect_pusht
```

The rollout video is written to `outputs/pusht_inspection/`.

Download and inspect the official Push-T demonstrations:

```bash
mkdir -p data
curl -fL https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip -o data/pusht.zip
echo "63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9  data/pusht.zip" | sha256sum --check
unzip -q data/pusht.zip -d data
uv run python -m src.tools.inspect_pusht_data
```

Dataset plots and an expert-episode video are written to `outputs/pusht_dataset/`.

Build the episode-safe `To=2`, `Tp=16`, `Ta=8` training and validation
dataloaders with:

```python
from src.pusht_dataset import create_pusht_dataloaders

loaders = create_pusht_dataloaders(batch_size=64)
batch = next(iter(loaders.train))
print(batch["observation"].shape)  # (64, 2, 5)
print(batch["action"].shape)       # (64, 16, 2)
```

Run the dataset tests with:

```bash
uv run pytest
```

Run a three-epoch offline sanity check with no DataLoader workers, W&B run, or
saved artifacts:

```bash
uv run python -m src.train.train_simple --sanity-check
```

Then train it on the complete training split:

```bash
uv run python -m src.train.train_simple
```

Train the same MLP as a small diffusion policy:

```bash
uv run python -m src.train.train_simple --policy diffusion
```

It predicts the Gaussian noise added to normalized expert action trajectories.
At inference time, DDPM sampling starts from a random `16 x 2` trajectory and
removes noise over 100 model calls.

Each training run gets a unique directory named after its W&B run, such as
`outputs/simple_bc/jumping-fog-3-vm2b1036/`. Its best-validation checkpoint,
final checkpoint, and loss history are saved as `best.pt`, `last.pt`, and
`history.json` inside that directory.
Training and validation curves are also logged to the
`shatz-visuomotor-diffusion` project in Weights & Biases. To run without
uploading, pass `--wandb-mode disabled`; use `--wandb-mode offline` to save a
run for later synchronization. By default, no random seeds are set. Pass
`--seed 42` to seed Python, NumPy, PyTorch, the episode split, and DataLoader
shuffling for a reproducible run.

Evaluate the best and last checkpoints from a run on identical Push-T seeds:

```bash
uv run python -m src.tools.evaluate_simple \
  outputs/simple_bc/<run>/best.pt \
  outputs/simple_bc/<run>/last.pt
```

For example, compare the diffusion `leafy-grass` run with the deterministic
`playful-meadow` run on the same 20 seeds:

```bash
uv run python -m src.tools.evaluate_simple \
  outputs/simple_bc/leafy-grass-6-6g8vzozm/best.pt \
  outputs/simple_bc/playful-meadow-5-6oh483cp/best.pt
```

The videos from that comparison are saved in:

```text
outputs/simple_evaluation/20260831-025115/leafy-grass-6-6g8vzozm/best/
outputs/simple_evaluation/20260831-025115/playful-meadow-5-6oh483cp/best/
```

The evaluator reports geometric target coverage, success above 95% coverage,
rollout length, and end-to-end policy inference latency. JSON results and a
video of every rollout are written to a timestamped directory under
`outputs/simple_evaluation/`. Each video displays current and maximum coverage
plus the 95% success threshold, and JSON results retain coverage at every step.
The evaluator automatically uses iterative DDPM sampling for diffusion
checkpoints, so deterministic and diffusion checkpoints can be compared with
the same command and rollout seeds.
