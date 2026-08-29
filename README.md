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
