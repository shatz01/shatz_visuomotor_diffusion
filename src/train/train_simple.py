"""Train simple MLP policies on Push-T action trajectories."""

from __future__ import annotations

import json

import torch
import wandb
from torch import nn

from src.pusht_dataset import PushTNormalizer, create_pusht_dataloaders
from src.train.args import parse_args
from src.train.cosine_beta_schedule import cosine_beta_schedule
from src.train.model import SimpleTrajectoryModel
from src.train.train_utils import (
    create_run_output_dir,
    resolve_device,
    save_checkpoint,
    set_seed,
)


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


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
