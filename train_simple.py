"""Train simple MLP policies on Push-T action trajectories."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn

from cosine_beta_schedule import cosine_beta_schedule
from pusht_dataset import (
    DEFAULT_DATASET,
    PushTNormalizer,
    create_pusht_dataloaders,
)


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class SimpleTrajectoryModel(nn.Module):
    """Predict an action trajectory directly or predict diffusion noise."""

    def __init__(
        self,
        observation_horizon: int = 2,
        observation_dim: int = 5,
        prediction_horizon: int = 16,
        action_dim: int = 2,
        hidden_dim: int = 256,
        residual_blocks: int = 4,
        using_diffusion_mode: bool = False,
        num_diffusion_steps: int = 100,
    ) -> None:
        super().__init__()
        self.observation_horizon = observation_horizon
        self.observation_dim = observation_dim
        self.prediction_horizon = prediction_horizon
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.using_diffusion_mode = using_diffusion_mode

        self.input_dim = observation_horizon * observation_dim
        if using_diffusion_mode:
            if num_diffusion_steps < 2:
                raise ValueError("Diffusion requires at least two timesteps.")
            self.input_dim += prediction_horizon * action_dim + 1
            # when doing diffusion, model needs to take observations + noised trajectories + noise steps used
            # using that, we predict noise used.

        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            *(ResidualBlock(hidden_dim) for _ in range(residual_blocks)),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, prediction_horizon * action_dim),
        )

    def forward(
        self,
        observation: torch.Tensor,
        noisy_actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.observation_horizon, self.observation_dim)
        if tuple(observation.shape[1:]) != expected:
            raise ValueError(
                f"Expected observation shape (batch, {expected[0]}, {expected[1]}), "
                f"received {tuple(observation.shape)}."
            )

        model_input = observation.flatten(start_dim=1)
        if self.using_diffusion_mode:
            if noisy_actions is None or timesteps is None:
                raise ValueError("Diffusion mode requires noisy actions and timesteps.")
            normalized_timesteps = (
                timesteps.float() / (self.num_diffusion_steps - 1)
            ).unsqueeze(1)
            model_input = torch.cat(
                [
                    model_input,
                    noisy_actions.flatten(start_dim=1),
                    normalized_timesteps,
                ],
                dim=1,
            )

        action = self.network(model_input)
        return action.reshape(-1, self.prediction_horizon, self.action_dim)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha_bars: torch.Tensor | None = None,
) -> float:
    model.train()
    squared_error = 0.0
    elements = 0
    for batch in loader:
        # 1 observation is 5 values: [agent_x, agent_y, block_x, block_y, block_angle]
        observation = batch["observation"].to(device, non_blocking=True)  # [B, 2, 5]
        target_clean_trajectory = batch["action"].to(device, non_blocking=True)  # [B, 16, 2]

        if alpha_bars is None:
            prediction = model(observation)
            target = target_clean_trajectory
        else:
            timesteps = torch.randint(
                0,
                len(alpha_bars),
                (observation.shape[0],),
                device=device,
            )
            target_noise = torch.randn_like(target_clean_trajectory)
            alpha_bar_t = alpha_bars[timesteps].view(-1, 1, 1)
            noisy_actions = (
                alpha_bar_t.sqrt() * target_clean_trajectory
                + (1 - alpha_bar_t).sqrt() * target_noise
            )
            prediction = model(observation, noisy_actions, timesteps)
            target = target_noise

        loss = nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        squared_error += float(loss.detach()) * target.numel()
        elements += target.numel()

    return squared_error / elements


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    normalizer: PushTNormalizer,
    execution_slice: slice,
    device: torch.device,
    alpha_bars: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    elements = 0
    trajectory_distance = 0.0
    execution_distance = 0.0
    action_steps = 0
    execution_steps = 0

    for batch in loader:
        observation = batch["observation"].to(device, non_blocking=True)
        target_clean_trajectory = batch["action"].to(device, non_blocking=True)

        if alpha_bars is None:
            prediction = model(observation)
            target = target_clean_trajectory
        else:
            timesteps = torch.randint(
                0,
                len(alpha_bars),
                (observation.shape[0],),
                device=device,
            )
            target_noise = torch.randn_like(target_clean_trajectory)
            alpha_bar_t = alpha_bars[timesteps].view(-1, 1, 1)
            noisy_actions = (
                alpha_bar_t.sqrt() * target_clean_trajectory
                + (1 - alpha_bar_t).sqrt() * target_noise
            )
            prediction = model(observation, noisy_actions, timesteps)
            target = target_noise

        squared_error += float(nn.functional.mse_loss(prediction, target)) * target.numel()
        elements += target.numel()

        if alpha_bars is not None:
            continue

        prediction_pixels = normalizer.action.denormalize(prediction)
        target_pixels = normalizer.action.denormalize(target_clean_trajectory)
        distances = torch.linalg.vector_norm(prediction_pixels - target_pixels, dim=-1)
        trajectory_distance += float(distances.sum())
        action_steps += distances.numel()
        execution_distance += float(distances[:, execution_slice].sum())
        execution_steps += distances[:, execution_slice].numel()

    metrics = {"mse": squared_error / elements}
    if alpha_bars is None:
        metrics.update(
            {
                "trajectory_error_px": trajectory_distance / action_steps,
                "execution_error_px": execution_distance / execution_steps,
            }
        )
    return metrics


def fit_fixed_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    steps: int,
    device: torch.device,
    alpha_bars: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Optimize one fixed batch as an end-to-end pipeline sanity check."""
    observation = batch["observation"].to(device)
    target_clean_trajectory = batch["action"].to(device)

    if alpha_bars is None:
        model_inputs = (observation,)
        target = target_clean_trajectory
    else:
        timesteps = torch.randint(
            0,
            len(alpha_bars),
            (observation.shape[0],),
            device=device,
        )
        target_noise = torch.randn_like(target_clean_trajectory)
        alpha_bar_t = alpha_bars[timesteps].view(-1, 1, 1)
        noisy_actions = (
            alpha_bar_t.sqrt() * target_clean_trajectory
            + (1 - alpha_bar_t).sqrt() * target_noise
        )
        model_inputs = (observation, noisy_actions, timesteps)
        target = target_noise

    model.train()
    with torch.no_grad():
        initial_loss = float(nn.functional.mse_loss(model(*model_inputs), target))

    for _ in range(steps):
        loss = nn.functional.mse_loss(model(*model_inputs), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = float(nn.functional.mse_loss(model(*model_inputs), target))
    return initial_loss, final_loss


def save_checkpoint(
    path: Path,
    model: SimpleTrajectoryModel,
    metadata: dict[str, object],
    epoch: int | None,
    validation_metrics: dict[str, float] | None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            **metadata,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
        },
        path,
    )


def create_run_output_dir(output_root: Path, run_name: str, run_id: str) -> Path:
    """Create a unique, filesystem-safe directory for one W&B run."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip("._-") or "run"
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "-", run_id).strip("._-")
    if not safe_id:
        raise ValueError("The W&B run ID cannot be empty.")
    output_dir = output_root / f"{safe_name}-{safe_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/simple_bc"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overfit-one-batch", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=1_000)
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument(
        "--wandb-project", default="shatz-visuomotor-diffusion"
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument(
        "--policy",
        choices=("simple", "diffusion"),
        default="simple",
    )
    parser.add_argument("--num-diffusion-steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sanity_check and args.overfit_one_batch:
        raise ValueError("--sanity-check and --overfit-one-batch cannot be combined.")
    if args.sanity_check:
        args.epochs = 3
        args.num_workers = 0
    if args.seed is not None:
        set_seed(args.seed)
    device = resolve_device(args.device)

    yes_diffusion = args.policy == "diffusion"
    num_diffusion_steps = args.num_diffusion_steps
    if yes_diffusion:
        print(f"✅ doing diffusion, num_steps: {num_diffusion_steps}")
        # betas are how much noise we add per step. 0th step very little noise. Last step almost all noise.
        betas = cosine_beta_schedule(num_diffusion_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0).to(device)
    else:
        alpha_bars = None
        print("❌ No diffusion")

    loaders = create_pusht_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    observation_dim = loaders.train.dataset[0]["observation"].shape[-1]
    model_config = {
        "observation_horizon": loaders.train.dataset.observation_horizon,
        "observation_dim": observation_dim,
        "prediction_horizon": loaders.train.dataset.prediction_horizon,
        "action_dim": 2,
        "hidden_dim": args.hidden_dim,
        "residual_blocks": args.residual_blocks,
        "using_diffusion_mode": yes_diffusion,
        "num_diffusion_steps": num_diffusion_steps,
    }
    model = SimpleTrajectoryModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    normalizer = loaders.normalizer
    checkpoint_metadata = {
        "model_config": model_config,
        "normalizer": {
            "observation_minimum": torch.from_numpy(normalizer.observation.minimum),
            "observation_maximum": torch.from_numpy(normalizer.observation.maximum),
            "action_minimum": torch.from_numpy(normalizer.action.minimum),
            "action_maximum": torch.from_numpy(normalizer.action.maximum),
        },
        "train_episodes": loaders.train_episodes,
        "validation_episodes": loaders.validation_episodes,
    }

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print(f"Random seed: {args.seed if args.seed is not None else 'not set'}")
    print(f"Parameters: {parameters:,}")
    print(
        f"Windows: {len(loaders.train.dataset):,} train / "
        f"{len(loaders.validation.dataset):,} validation"
    )
    run = None
    run_output_dir = None
    if args.sanity_check:
        print("Sanity check: 3 epochs, 0 workers, no logging or saved artifacts")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            dir=str(args.output_dir),
            settings=wandb.Settings(x_disable_stats=True),
            config={
                "model": "simple_residual_mlp",
                "dataset": str(args.dataset),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "device": str(device),
                "parameters": parameters,
                "train_windows": len(loaders.train.dataset),
                "validation_windows": len(loaders.validation.dataset),
                **model_config,
            },
        )
        run_output_dir = create_run_output_dir(args.output_dir, run.name, run.id)
        run.config.update({"local_output_dir": str(run_output_dir)})
        print(f"Run: {run.name} ({run.id})")

    if args.overfit_one_batch:
        batch = next(iter(loaders.train))
        initial_loss, final_loss = fit_fixed_batch(
            model,
            batch,
            optimizer,
            args.overfit_steps,
            device,
            alpha_bars,
        )
        print(
            f"Fixed-batch MSE after {args.overfit_steps:,} steps: "
            f"{initial_loss:.6f} -> {final_loss:.6f}"
        )
        save_checkpoint(
            run_output_dir / "overfit.pt",
            model,
            checkpoint_metadata,
            epoch=None,
            validation_metrics=None,
        )
        run.log(
            {
                "overfit/initial_mse": initial_loss,
                "overfit/final_mse": final_loss,
            }
        )
        run.summary["overfit_initial_mse"] = initial_loss
        run.summary["overfit_final_mse"] = final_loss
        run.finish()
        return

    best_mse = float("inf")
    best_epoch = 0
    history = []
    execution_slice = loaders.train.dataset.execution_slice
    for epoch in range(1, args.epochs + 1):
        train_mse = train_epoch(model, loaders.train, optimizer, device, alpha_bars=alpha_bars)
        validation = evaluate(
            model,
            loaders.validation,
            loaders.normalizer,
            execution_slice,
            device,
            alpha_bars,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train/mse": train_mse,
            **{f"validation/{name}": value for name, value in validation.items()},
        }
        history.append(epoch_metrics)
        if run is not None:
            run.log(epoch_metrics, step=epoch)
        epoch_message = (
            f"Epoch {epoch:03d} | train MSE {train_mse:.6f} | "
            f"validation MSE {validation['mse']:.6f}"
        )
        if "execution_error_px" in validation:
            epoch_message += (
                f" | execution error {validation['execution_error_px']:.2f}px"
            )
        print(epoch_message)
        if validation["mse"] < best_mse:
            best_mse = validation["mse"]
            best_epoch = epoch
            if run_output_dir is not None:
                save_checkpoint(
                    run_output_dir / "best.pt",
                    model,
                    checkpoint_metadata,
                    epoch,
                    validation,
                )
        if run_output_dir is not None:
            save_checkpoint(
                run_output_dir / "last.pt",
                model,
                checkpoint_metadata,
                epoch,
                validation,
            )
            (run_output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )

    print(f"Best validation MSE: {best_mse:.6f}")
    if run is not None:
        print(f"Artifacts: {run_output_dir.resolve()}")
        run.summary["best_validation_mse"] = best_mse
        run.summary["best_epoch"] = best_epoch
        run.finish()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


if __name__ == "__main__":
    main()
