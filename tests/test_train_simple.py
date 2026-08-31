import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.train.diffusion import sample_actions
from src.train.model import SimpleTrajectoryModel
from src.train.train_utils import create_run_output_dir, set_seed


def test_model_shape_and_gradients() -> None:
    model = SimpleTrajectoryModel(hidden_dim=32, residual_blocks=2)
    observation = torch.randn(4, 2, 5)
    target = torch.randn(4, 16, 2)
    prediction = model(observation)

    assert prediction.shape == target.shape
    torch.nn.functional.mse_loss(prediction, target).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_diffusion_model_shape_and_gradients() -> None:
    model = SimpleTrajectoryModel(
        hidden_dim=32,
        residual_blocks=2,
        using_diffusion_mode=True,
        num_diffusion_steps=100,
    )
    observation = torch.randn(4, 2, 5)
    noisy_actions = torch.randn(4, 16, 2)
    timesteps = torch.randint(0, 100, (4,))
    target_noise = torch.randn(4, 16, 2)
    prediction = model(observation, noisy_actions, timesteps)

    assert prediction.shape == target_noise.shape
    torch.nn.functional.mse_loss(prediction, target_noise).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_reverse_diffusion_sampling_shape() -> None:
    model = SimpleTrajectoryModel(
        hidden_dim=32,
        residual_blocks=1,
        using_diffusion_mode=True,
        num_diffusion_steps=4,
    )
    sampled_actions = sample_actions(model, torch.randn(2, 2, 5))

    assert sampled_actions.shape == (2, 16, 2)
    assert torch.isfinite(sampled_actions).all()


def test_model_rejects_wrong_observation_shape() -> None:
    model = SimpleTrajectoryModel()
    try:
        model(torch.randn(4, 5))
    except ValueError as error:
        assert "Expected observation shape" in str(error)
    else:
        raise AssertionError("Expected the model to reject an invalid input shape.")


def test_run_output_directory_uses_safe_wandb_identity(tmp_path: Path) -> None:
    output_dir = create_run_output_dir(tmp_path, "helpful run/name", "abc123")
    assert output_dir == tmp_path / "helpful-run-name-abc123"
    assert output_dir.is_dir()
    with pytest.raises(FileExistsError):
        create_run_output_dir(tmp_path, "helpful run/name", "abc123")


def test_explicit_seed_controls_all_random_generators() -> None:
    set_seed(7)
    first = (random.random(), np.random.random(), torch.rand(1))
    set_seed(7)
    second = (random.random(), np.random.random(), torch.rand(1))
    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])
