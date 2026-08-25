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
