from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tools.evaluate_simple import (
    RolloutResult,
    annotate_frame,
    load_policy,
    predict_action_chunk,
    summarize,
)
from src.pusht_dataset import MinMaxStats, PushTNormalizer
from src.train.model import SimpleTrajectoryModel


def test_load_policy_and_predict_execution_chunk(tmp_path: Path) -> None:
    model = SimpleTrajectoryModel(hidden_dim=16, residual_blocks=1)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint = {
        "model": model.state_dict(),
        "model_config": {
            "observation_horizon": 2,
            "observation_dim": 5,
            "prediction_horizon": 16,
            "action_dim": 2,
            "hidden_dim": 16,
            "residual_blocks": 1,
        },
        "normalizer": {
            "observation_minimum": torch.zeros(5),
            "observation_maximum": torch.full((5,), 10.0),
            "action_minimum": torch.zeros(2),
            "action_maximum": torch.full((2,), 512.0),
        },
        "epoch": 3,
        "validation_metrics": {"mse": 0.1},
    }
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)

    loaded, normalizer, loaded_checkpoint = load_policy(path, torch.device("cpu"))
    history = deque([np.zeros(5), np.ones(5)], maxlen=2)
    actions, latency = predict_action_chunk(
        loaded, normalizer, history, action_horizon=8, device=torch.device("cpu")
    )

    assert loaded_checkpoint["epoch"] == 3
    assert actions.shape == (8, 2)
    np.testing.assert_allclose(actions, 256.0)
    assert latency >= 0


def test_summarize_rollouts() -> None:
    results = [
        RolloutResult(0, True, 100, 13, 0.96, 0.96, 1.0, [0.0, 0.96]),
        RolloutResult(1, False, 300, 38, 0.50, 0.40, 3.0, [0.0, 0.40]),
    ]
    summary = summarize(results)
    assert summary["episodes"] == 2
    assert summary["successes"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["mean_maximum_coverage"] == pytest.approx(0.73)
    assert summary["mean_final_coverage"] == pytest.approx(0.68)
    assert summary["mean_steps"] == 200
    assert summary["mean_inference_ms"] == 2.0


def test_coverage_annotation_preserves_frame() -> None:
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    annotated = annotate_frame(
        frame, step=10, coverage=0.96, maximum_coverage=0.96, success=True
    )
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert np.any(annotated != frame)
