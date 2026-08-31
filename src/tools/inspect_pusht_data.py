"""Inspect the official Push-T demonstration dataset."""

import argparse
from pathlib import Path

import gym_pusht  # noqa: F401: importing registers the Push-T environment
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import zarr
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


DEFAULT_DATASET = Path("data/pusht/pusht_cchi_v7_replay.zarr")
DEFAULT_OUTPUT = Path("outputs/pusht_dataset")
CONTROL_HZ = 10


def summarize_array(array: zarr.Array, batch_size: int = 512) -> str:
    """Compute scalar statistics without loading a large array all at once."""
    count = 0
    total = 0.0
    total_squared = 0.0
    minimum = np.inf
    maximum = -np.inf
    all_finite = True

    for start in range(0, array.shape[0], batch_size):
        batch = np.asarray(array[start : start + batch_size], dtype=np.float64)
        all_finite &= bool(np.isfinite(batch).all())
        count += batch.size
        total += float(batch.sum())
        total_squared += float(np.square(batch).sum())
        minimum = min(minimum, float(batch.min()))
        maximum = max(maximum, float(batch.max()))

    mean = total / count
    variance = max(total_squared / count - mean**2, 0.0)
    return (
        f"shape={array.shape}, dtype={array.dtype}, finite={all_finite}, "
        f"min={minimum:.3f}, max={maximum:.3f}, "
        f"mean={mean:.3f}, std={variance**0.5:.3f}"
    )


def save_episode_artifacts(
    data: zarr.Group,
    start: int,
    end: int,
    episode: int,
    output_dir: Path,
) -> None:
    state = np.asarray(data["state"][start:end])
    action = np.asarray(data["action"][start:end])
    images = np.asarray(data["img"][start:end]).clip(0, 255).astype(np.uint8)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.plot(state[:, 0], state[:, 1], label="agent position", linewidth=2)
    axis.plot(action[:, 0], action[:, 1], label="action target", alpha=0.7)
    axis.plot(state[:, 2], state[:, 3], label="block center", linewidth=2)
    axis.scatter(*state[0, :2], label="agent start", marker="o")
    axis.scatter(*state[0, 2:4], label="block start", marker="s")
    axis.set(xlim=(0, 512), ylim=(512, 0), xlabel="x", ylabel="y")
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / f"episode_{episode:03d}_trajectories.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    video_path = output_dir / f"episode_{episode:03d}_expert.mp4"
    clip = ImageSequenceClip(list(images), fps=CONTROL_HZ)
    clip.write_videofile(
        str(video_path), codec="libx264", audio=False, logger=None
    )
    clip.close()

    print(f"Trajectory plot: {plot_path.resolve()}")
    print(f"Expert video:    {video_path.resolve()}")


def replay_episode(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Replay actions in gym-pusht from the recorded initial state."""
    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    initial_state = states[0].astype(np.float64)

    # gym-pusht intentionally preserves a legacy reset convention where setting
    # the block angle shifts its reported center. Measure and remove that shift.
    trial_state, _ = env.reset(options={"reset_to_state": initial_state})
    corrected_state = initial_state.copy()
    corrected_state[2:4] -= trial_state[2:4] - initial_state[2:4]
    observation, _ = env.reset(options={"reset_to_state": corrected_state})

    replayed_states = [observation]
    for action in actions[:-1]:
        observation, _, terminated, truncated, _ = env.step(action)
        replayed_states.append(observation)
        if terminated or truncated:
            break

    env.close()
    return np.asarray(replayed_states)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    root = zarr.open_group(str(args.dataset), mode="r")
    data = root["data"]
    episode_ends = np.asarray(root["meta"]["episode_ends"])
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    episode_lengths = episode_ends - episode_starts

    if not 0 <= args.episode < len(episode_ends):
        raise SystemExit(
            f"Episode must be between 0 and {len(episode_ends) - 1}."
        )

    print(f"Dataset: {args.dataset.resolve()}")
    print(f"Episodes: {len(episode_ends)}")
    print(f"Transitions: {episode_ends[-1]}")
    print(
        "Episode lengths: "
        f"min={episode_lengths.min()}, mean={episode_lengths.mean():.1f}, "
        f"median={np.median(episode_lengths):.1f}, max={episode_lengths.max()}"
    )
    print("\nArrays:")
    for name in sorted(data.array_keys()):
        print(f"  {name:10s} {summarize_array(data[name])}")

    state = np.asarray(data["state"])
    action = np.asarray(data["action"])
    within_action_bounds = bool(((action >= 0) & (action <= 512)).all())

    valid = np.ones(len(state) - 1, dtype=bool)
    valid[episode_ends[:-1] - 1] = False
    current_position = state[:-1, :2][valid]
    next_position = state[1:, :2][valid]
    current_action = action[:-1][valid]
    current_distance = np.linalg.norm(current_action - current_position, axis=1)
    next_distance = np.linalg.norm(current_action - next_position, axis=1)
    closer_fraction = float((next_distance < current_distance).mean())

    print("\nCompatibility checks:")
    print(f"  Episode ends are strictly increasing: {bool(np.all(np.diff(episode_ends) > 0))}")
    print(f"  Actions lie inside gym-pusht [0, 512] bounds: {within_action_bounds}")
    print(
        "  Next agent position moves closer to current action: "
        f"{closer_fraction:.3%} of transitions"
    )
    print("  State convention: [agent_x, agent_y, block_x, block_y, block_angle]")
    print("  Action convention: absolute [target_x, target_y] position")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = int(episode_starts[args.episode])
    end = int(episode_ends[args.episode])
    print(f"\nSelected episode: {args.episode} ({end - start} steps)")
    episode_states = state[start:end]
    episode_actions = action[start:end]
    replayed_states = replay_episode(episode_states, episode_actions)
    reference_states = episode_states[: len(replayed_states)]
    agent_error = np.linalg.norm(
        replayed_states[:, :2] - reference_states[:, :2], axis=1
    )
    block_error = np.linalg.norm(
        replayed_states[:, 2:4] - reference_states[:, 2:4], axis=1
    )
    angle_delta = replayed_states[:, 4] - reference_states[:, 4]
    angle_error = np.abs(np.arctan2(np.sin(angle_delta), np.cos(angle_delta)))
    print(
        "Simulator replay mean error: "
        f"agent={agent_error.mean():.2e}px, "
        f"block={block_error.mean():.2e}px, angle={angle_error.mean():.2e}rad"
    )
    save_episode_artifacts(data, start, end, args.episode, args.output_dir)


if __name__ == "__main__":
    main()
