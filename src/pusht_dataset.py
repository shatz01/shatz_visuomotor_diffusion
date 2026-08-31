"""Episode-safe Push-T sequence datasets and PyTorch dataloaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET = Path("data/pusht/pusht_cchi_v7_replay.zarr")
ObservationType = Literal["state", "keypoints"]


@dataclass(frozen=True)
class SequenceIndex:
    """Locations needed to extract and edge-pad one temporal window."""

    episode: int
    buffer_start: int
    buffer_end: int
    sample_start: int
    sample_end: int


@dataclass(frozen=True)
class MinMaxStats:
    """Per-feature extrema for mapping values to and from [-1, 1]."""

    minimum: np.ndarray
    maximum: np.ndarray

    def normalize(self, value: np.ndarray | torch.Tensor):
        minimum, scale = self._parameters_for(value)
        return (value - minimum) / scale * 2 - 1

    def denormalize(self, value: np.ndarray | torch.Tensor):
        minimum, scale = self._parameters_for(value)
        return (value + 1) / 2 * scale + minimum

    def _parameters_for(self, value: np.ndarray | torch.Tensor):
        scale = np.maximum(self.maximum - self.minimum, 1e-8)
        if isinstance(value, torch.Tensor):
            minimum = torch.as_tensor(
                self.minimum, dtype=value.dtype, device=value.device
            )
            scale = torch.as_tensor(scale, dtype=value.dtype, device=value.device)
            return minimum, scale
        return self.minimum, scale


@dataclass(frozen=True)
class PushTNormalizer:
    observation: MinMaxStats
    action: MinMaxStats


@dataclass(frozen=True)
class PushTDataLoaders:
    train: DataLoader
    validation: DataLoader
    normalizer: PushTNormalizer
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]


def split_episodes(
    number_of_episodes: int,
    validation_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Make an episode-level split with no shared transitions."""
    if number_of_episodes < 2:
        raise ValueError("At least two episodes are required for a split.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(number_of_episodes)
    validation_count = min(
        max(round(number_of_episodes * validation_fraction), 1),
        number_of_episodes - 1,
    )
    validation = tuple(sorted(int(i) for i in shuffled[:validation_count]))
    train = tuple(sorted(int(i) for i in shuffled[validation_count:]))
    return train, validation


def make_sequence_indices(
    episode_ends: np.ndarray,
    episodes: Sequence[int],
    prediction_horizon: int,
    observation_horizon: int,
    action_horizon: int,
) -> list[SequenceIndex]:
    """Create Diffusion Policy windows without crossing episode boundaries."""
    if min(prediction_horizon, observation_horizon, action_horizon) < 1:
        raise ValueError("All horizons must be positive.")
    if observation_horizon + action_horizon - 1 > prediction_horizon:
        raise ValueError(
            "prediction_horizon must fit the observation offset and action horizon."
        )

    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[SequenceIndex] = []
    pad_before = observation_horizon - 1
    pad_after = action_horizon - 1

    for episode in episodes:
        episode_start = int(starts[episode])
        episode_end = int(episode_ends[episode])
        episode_length = episode_end - episode_start
        minimum_start = -pad_before
        maximum_start = episode_length - prediction_horizon + pad_after

        for relative_start in range(minimum_start, maximum_start + 1):
            buffer_start = episode_start + max(relative_start, 0)
            buffer_end = episode_start + min(
                relative_start + prediction_horizon, episode_length
            )
            sample_start = max(-relative_start, 0)
            sample_end = sample_start + buffer_end - buffer_start
            indices.append(
                SequenceIndex(
                    episode=episode,
                    buffer_start=buffer_start,
                    buffer_end=buffer_end,
                    sample_start=sample_start,
                    sample_end=sample_end,
                )
            )
    return indices


class PushTSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length normalized action trajectories conditioned on observations.

    A sample contains ``observation_horizon`` observations and
    ``prediction_horizon`` actions. To execute the current/future chunk, slice
    actions from ``observation_horizon - 1`` for ``action_horizon`` entries.
    Missing values at episode boundaries repeat the nearest real value, matching
    the sequence construction used by Diffusion Policy.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        episodes: Sequence[int],
        normalizer: PushTNormalizer,
        observation_type: ObservationType = "state",
        observation_horizon: int = 2,
        prediction_horizon: int = 16,
        action_horizon: int = 8,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.episodes = tuple(int(episode) for episode in episodes)
        self.normalizer = normalizer
        self.observation_type = observation_type
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.action_horizon = action_horizon
        self.execution_slice = slice(
            observation_horizon - 1,
            observation_horizon - 1 + action_horizon,
        )
        self._root: zarr.Group | None = None

        root = zarr.open_group(str(self.dataset_path), mode="r")
        episode_ends = np.asarray(root["meta/episode_ends"])
        if not self.episodes:
            raise ValueError("episodes cannot be empty.")
        if min(self.episodes) < 0 or max(self.episodes) >= len(episode_ends):
            raise IndexError("An episode index is outside the dataset.")
        self.indices = make_sequence_indices(
            episode_ends,
            self.episodes,
            prediction_horizon,
            observation_horizon,
            action_horizon,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        location = self.indices[index]
        data = self._data
        observation = _read_observations(
            data,
            location.buffer_start,
            location.buffer_end,
            self.observation_type,
        )
        action = np.asarray(
            data["action"][location.buffer_start : location.buffer_end],
            dtype=np.float32,
        )
        observation = _edge_pad(
            observation,
            self.prediction_horizon,
            location.sample_start,
            location.sample_end,
        )
        action = _edge_pad(
            action,
            self.prediction_horizon,
            location.sample_start,
            location.sample_end,
        )
        observation = observation[: self.observation_horizon]

        return {
            "observation": torch.from_numpy(
                self.normalizer.observation.normalize(observation).astype(np.float32)
            ),
            "action": torch.from_numpy(
                self.normalizer.action.normalize(action).astype(np.float32)
            ),
        }

    @property
    def _data(self) -> zarr.Group:
        if self._root is None:
            self._root = zarr.open_group(str(self.dataset_path), mode="r")
        return self._root["data"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_root"] = None
        return state


def create_pusht_dataloaders(
    dataset_path: Path | str = DEFAULT_DATASET,
    *,
    batch_size: int = 64,
    validation_fraction: float = 0.2,
    seed: int | None = None,
    observation_type: ObservationType = "state",
    observation_horizon: int = 2,
    prediction_horizon: int = 16,
    action_horizon: int = 8,
    num_workers: int = 0,
) -> PushTDataLoaders:
    """Build loaders, using ``seed`` for both splitting and shuffling if set."""
    dataset_path = Path(dataset_path)
    root = zarr.open_group(str(dataset_path), mode="r")
    episode_ends = np.asarray(root["meta/episode_ends"])
    train_episodes, validation_episodes = split_episodes(
        len(episode_ends), validation_fraction, seed
    )
    normalizer = fit_normalizer(
        root["data"], episode_ends, train_episodes, observation_type
    )

    common = dict(
        dataset_path=dataset_path,
        normalizer=normalizer,
        observation_type=observation_type,
        observation_horizon=observation_horizon,
        prediction_horizon=prediction_horizon,
        action_horizon=action_horizon,
    )
    train_dataset = PushTSequenceDataset(episodes=train_episodes, **common)
    validation_dataset = PushTSequenceDataset(episodes=validation_episodes, **common)
    loader_options = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    generator = (
        torch.Generator().manual_seed(seed) if seed is not None else None
    )
    return PushTDataLoaders(
        train=DataLoader(
            train_dataset, shuffle=True, generator=generator, **loader_options
        ),
        validation=DataLoader(validation_dataset, shuffle=False, **loader_options),
        normalizer=normalizer,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
    )


def fit_normalizer(
    data: Mapping[str, zarr.Array],
    episode_ends: np.ndarray,
    episodes: Sequence[int],
    observation_type: ObservationType,
) -> PushTNormalizer:
    """Fit feature-wise extrema using raw transitions from selected episodes."""
    starts = np.concatenate(([0], episode_ends[:-1]))
    observations = []
    actions = []
    for episode in episodes:
        start, end = int(starts[episode]), int(episode_ends[episode])
        observations.append(_read_observations(data, start, end, observation_type))
        actions.append(np.asarray(data["action"][start:end], dtype=np.float32))
    return PushTNormalizer(
        observation=_fit_stats(np.concatenate(observations)),
        action=_fit_stats(np.concatenate(actions)),
    )


def _read_observations(
    data: Mapping[str, zarr.Array],
    start: int,
    end: int,
    observation_type: ObservationType,
) -> np.ndarray:
    state = np.asarray(data["state"][start:end], dtype=np.float32)
    if observation_type == "state":
        return state
    if observation_type == "keypoints":
        keypoints = np.asarray(data["keypoint"][start:end], dtype=np.float32)
        return np.concatenate((keypoints.reshape(len(keypoints), -1), state[:, :2]), axis=-1)
    raise ValueError(f"Unknown observation_type: {observation_type}")


def _fit_stats(values: np.ndarray) -> MinMaxStats:
    flattened = values.reshape(-1, values.shape[-1])
    return MinMaxStats(flattened.min(axis=0), flattened.max(axis=0))


def _edge_pad(
    values: np.ndarray,
    length: int,
    sample_start: int,
    sample_end: int,
) -> np.ndarray:
    if sample_start == 0 and sample_end == length:
        return values
    padded = np.empty((length, *values.shape[1:]), dtype=values.dtype)
    padded[:sample_start] = values[0]
    padded[sample_start:sample_end] = values
    padded[sample_end:] = values[-1]
    return padded
