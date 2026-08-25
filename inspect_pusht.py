# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gym-pusht==0.1.6",
#     "moviepy>=2.0,<3",
#     "pymunk>=6.6,<7",
# ]
# ///

"""Inspect Push-T and save a deterministic random-policy rollout."""

from pathlib import Path

import gym_pusht  # noqa: F401: importing registers the Push-T environment
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


SEED = 42
MAX_STEPS = 300
VIDEO_DIR = Path("outputs/pusht_inspection")


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
        max_episode_steps=MAX_STEPS,
    )
    env = RecordVideo(
        env,
        video_folder=str(VIDEO_DIR),
        name_prefix="random-policy",
        episode_trigger=lambda _: True,
        disable_logger=True,
    )

    observation, info = env.reset(seed=SEED)
    env.action_space.seed(SEED)

    print("Push-T environment")
    print(f"  observation space: {env.observation_space}")
    print(f"  action space:      {env.action_space}")
    print(f"  initial state:     {observation}")
    print(f"  info fields:       {sorted(info)}")

    max_reward = 0.0
    steps = 0
    success = False

    for steps in range(1, MAX_STEPS + 1):
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        max_reward = max(max_reward, float(reward))
        success = bool(info["is_success"])
        if terminated or truncated:
            break

    env.close()

    print(f"  rollout steps:     {steps}")
    print(f"  maximum coverage:  {max_reward:.3f}")
    print(f"  success:           {success}")
    print(f"  video directory:   {VIDEO_DIR.resolve()}")


if __name__ == "__main__":
    main()
