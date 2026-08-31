"""Evaluate trajectory-policy checkpoints in closed-loop Push-T."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import cv2
import gym_pusht  # noqa: F401: importing registers the Push-T environment
import gymnasium as gym
import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from src.pusht_dataset import MinMaxStats, PushTNormalizer
from src.train.model import SimpleTrajectoryModel
from src.train.train_utils import resolve_device


CONTROL_HZ = 10


@dataclass(frozen=True)
class RolloutResult:
    seed: int
    success: bool
    steps: int
    plans: int
    maximum_coverage: float
    final_coverage: float
    mean_inference_ms: float
    coverage_history: list[float]


def annotate_frame(
    frame: np.ndarray,
    *,
    step: int,
    coverage: float,
    maximum_coverage: float,
    success: bool,
) -> np.ndarray:
    """Overlay raw overlap coverage and the 95% success threshold."""
    frame = frame.copy()
    panel = frame.copy()
    panel_width = min(330, frame.shape[1] - 20)
    cv2.rectangle(panel, (10, 10), (10 + panel_width, 88), (255, 255, 255), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)

    status = " | SUCCESS" if success else ""
    cv2.putText(
        frame,
        f"Step {step} | Coverage {coverage:.1%}{status}",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 100, 0) if success else (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Maximum {maximum_coverage:.1%}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    bar_start, bar_end, bar_y = 20, 20 + panel_width - 20, 76
    cv2.line(frame, (bar_start, bar_y), (bar_end, bar_y), (170, 170, 170), 8)
    fill_end = bar_start + round((bar_end - bar_start) * min(coverage, 1.0))
    cv2.line(frame, (bar_start, bar_y), (fill_end, bar_y), (40, 160, 40), 8)
    threshold_x = bar_start + round((bar_end - bar_start) * 0.95)
    cv2.line(frame, (threshold_x, bar_y - 7), (threshold_x, bar_y + 7), (220, 30, 30), 2)
    return frame


def load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[SimpleTrajectoryModel, PushTNormalizer, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SimpleTrajectoryModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    stats = checkpoint["normalizer"]
    normalizer = PushTNormalizer(
        observation=MinMaxStats(
            stats["observation_minimum"].numpy(),
            stats["observation_maximum"].numpy(),
        ),
        action=MinMaxStats(
            stats["action_minimum"].numpy(),
            stats["action_maximum"].numpy(),
        ),
    )
    return model, normalizer, checkpoint


@torch.inference_mode()
def predict_action_chunk(
    model: SimpleTrajectoryModel,
    normalizer: PushTNormalizer,
    observation_history: deque[np.ndarray],
    action_horizon: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    observation = torch.as_tensor(
        np.stack(observation_history), dtype=torch.float32, device=device
    ).unsqueeze(0)
    observation = normalizer.observation.normalize(observation)
    prediction = model(observation)
    prediction = normalizer.action.denormalize(prediction)[0].cpu().numpy()
    start_index = model.observation_horizon - 1
    action_chunk = np.clip(
        prediction[start_index : start_index + action_horizon], 0, 512
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - start) * 1_000
    return action_chunk, inference_ms


def rollout(
    model: SimpleTrajectoryModel,
    normalizer: PushTNormalizer,
    *,
    seed: int,
    action_horizon: int,
    max_steps: int,
    video_path: Path,
    video_size: int,
    device: torch.device,
) -> RolloutResult:
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        visualization_width=video_size,
        visualization_height=video_size,
    )
    observation, _ = env.reset(seed=seed)
    observation_history = deque(
        [observation.copy() for _ in range(model.observation_horizon)],
        maxlen=model.observation_horizon,
    )
    initial_coverage = float(env.unwrapped._get_coverage())
    maximum_coverage = initial_coverage
    final_coverage = initial_coverage
    coverage_history = [initial_coverage]
    frames = [
        annotate_frame(
            env.render(),
            step=0,
            coverage=initial_coverage,
            maximum_coverage=initial_coverage,
            success=False,
        )
    ]
    inference_times = []
    success = False
    steps = 0

    while steps < max_steps and not success:
        actions, inference_ms = predict_action_chunk(
            model,
            normalizer,
            observation_history,
            action_horizon,
            device,
        )
        inference_times.append(inference_ms)
        for action in actions:
            observation, _, terminated, truncated, info = env.step(action)
            observation_history.append(observation)
            steps += 1
            final_coverage = float(info["coverage"])
            coverage_history.append(final_coverage)
            maximum_coverage = max(maximum_coverage, final_coverage)
            success = bool(info["is_success"])
            frames.append(
                annotate_frame(
                    env.render(),
                    step=steps,
                    coverage=final_coverage,
                    maximum_coverage=maximum_coverage,
                    success=success,
                )
            )
            if terminated or truncated:
                break

    env.close()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    clip = ImageSequenceClip(frames, fps=CONTROL_HZ)
    clip.write_videofile(
        str(video_path), codec="libx264", audio=False, logger=None
    )
    clip.close()

    return RolloutResult(
        seed=seed,
        success=success,
        steps=steps,
        plans=len(inference_times),
        maximum_coverage=maximum_coverage,
        final_coverage=final_coverage,
        mean_inference_ms=mean(inference_times),
        coverage_history=coverage_history,
    )


def summarize(results: list[RolloutResult]) -> dict[str, float | int]:
    return {
        "episodes": len(results),
        "successes": sum(result.success for result in results),
        "success_rate": mean(result.success for result in results),
        "mean_maximum_coverage": mean(
            result.maximum_coverage for result in results
        ),
        "mean_final_coverage": mean(result.final_coverage for result in results),
        "mean_steps": mean(result.steps for result in results),
        "mean_inference_ms": mean(result.mean_inference_ms for result in results),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    model, normalizer, checkpoint = load_policy(checkpoint_path, device)
    if args.action_horizon + model.observation_horizon - 1 > model.prediction_horizon:
        raise ValueError("The action horizon does not fit the predicted trajectory.")

    # Warm up CUDA before measuring inference latency.
    history = deque(
        [np.zeros(model.observation_dim, dtype=np.float32)]
        * model.observation_horizon,
        maxlen=model.observation_horizon,
    )
    predict_action_chunk(model, normalizer, history, args.action_horizon, device)

    results = []
    checkpoint_output = output_dir / checkpoint_path.parent.name / checkpoint_path.stem
    for episode in range(args.episodes):
        seed = args.seed_start + episode
        result = rollout(
            model,
            normalizer,
            seed=seed,
            action_horizon=args.action_horizon,
            max_steps=args.max_steps,
            video_path=checkpoint_output / f"seed_{seed:04d}.mp4",
            video_size=args.video_size,
            device=device,
        )
        results.append(result)
        print(
            f"  seed {seed:4d} | coverage {result.maximum_coverage:.3f} | "
            f"success {result.success} | steps {result.steps}"
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_metrics": checkpoint["validation_metrics"],
        "action_horizon": args.action_horizon,
        "summary": summarize(results),
        "rollouts": [asdict(result) for result in results],
    }
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    (checkpoint_output / "results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/simple_evaluation")
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--video-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True)
    print(f"Device: {device}")
    print(f"Evaluation output: {output_dir.resolve()}")

    reports = []
    for checkpoint_path in args.checkpoints:
        print(f"\nCheckpoint: {checkpoint_path}")
        reports.append(evaluate_checkpoint(checkpoint_path, output_dir, args, device))

    evaluation_config = {
        **vars(args),
        "checkpoints": [str(path) for path in args.checkpoints],
        "output_dir": str(args.output_dir),
    }
    comparison = {"evaluation": evaluation_config, "reports": reports}
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )

    print("\nSummary:")
    for report in reports:
        summary = report["summary"]
        print(
            f"  {Path(report['checkpoint']).name:8s} | "
            f"success {summary['success_rate']:.1%} | "
            f"max coverage {summary['mean_maximum_coverage']:.3f} | "
            f"inference {summary['mean_inference_ms']:.2f}ms"
        )


if __name__ == "__main__":
    main()
