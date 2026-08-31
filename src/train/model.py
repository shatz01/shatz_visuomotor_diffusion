"""MLP policy model used by simple regression and diffusion training."""

import torch
from torch import nn


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
