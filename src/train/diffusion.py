"""Reverse DDPM sampling for the simple trajectory model."""

import torch

from src.train.cosine_beta_schedule import cosine_beta_schedule
from src.train.model import SimpleTrajectoryModel


@torch.inference_mode()
def sample_actions(
    model: SimpleTrajectoryModel,
    observation: torch.Tensor,
) -> torch.Tensor:
    """Turn Gaussian noise into a normalized action trajectory."""
    betas = cosine_beta_schedule(model.num_diffusion_steps).to(observation.device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_previous = torch.cat(
        (torch.ones(1, device=observation.device), alpha_bars[:-1])
    )

    actions = torch.randn(
        observation.shape[0],
        model.prediction_horizon,
        model.action_dim,
        device=observation.device,
    )

    for step in reversed(range(model.num_diffusion_steps)):
        timesteps = torch.full(
            (observation.shape[0],),
            step,
            device=observation.device,
            dtype=torch.long,
        )
        predicted_noise = model(observation, actions, timesteps)

        predicted_clean_actions = (
            actions - (1 - alpha_bars[step]).sqrt() * predicted_noise
        ) / alpha_bars[step].sqrt()
        predicted_clean_actions = predicted_clean_actions.clamp(-1, 1)

        clean_weight = (
            betas[step] * alpha_bars_previous[step].sqrt()
            / (1 - alpha_bars[step])
        )
        noisy_weight = (
            alphas[step].sqrt()
            * (1 - alpha_bars_previous[step])
            / (1 - alpha_bars[step])
        )
        actions = clean_weight * predicted_clean_actions + noisy_weight * actions

        if step > 0:
            variance = (
                betas[step]
                * (1 - alpha_bars_previous[step])
                / (1 - alpha_bars[step])
            )
            actions += variance.sqrt() * torch.randn_like(actions)

    return actions
