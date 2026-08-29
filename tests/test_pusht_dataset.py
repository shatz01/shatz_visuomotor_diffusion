from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from pusht_dataset import (
    MinMaxStats,
    PushTSequenceDataset,
    create_pusht_dataloaders,
    make_sequence_indices,
)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.zarr"
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    episode_ends = np.array([5, 11, 18], dtype=np.int64)
    transition = np.arange(18, dtype=np.float32)
    state = np.stack([transition + offset for offset in range(5)], axis=1)
    action = np.stack([transition, -transition], axis=1)
    keypoint = np.repeat(transition[:, None, None], 18, axis=1).reshape(18, 9, 2)
    data.create_dataset("state", data=state)
    data.create_dataset("action", data=action)
    data.create_dataset("keypoint", data=keypoint)
    meta.create_dataset("episode_ends", data=episode_ends)
    return path


def test_sequence_indices_stay_inside_their_episode() -> None:
    ends = np.array([5, 11, 18])
    indices = make_sequence_indices(ends, [0, 2], 4, 2, 2)
    starts = np.array([0, 5, 11])
    assert all(
        starts[item.episode] <= item.buffer_start < item.buffer_end <= ends[item.episode]
        for item in indices
    )
    assert {item.episode for item in indices} == {0, 2}


def test_boundary_padding_and_action_alignment(tiny_dataset: Path) -> None:
    loaders = create_pusht_dataloaders(
        tiny_dataset,
        batch_size=2,
        validation_fraction=1 / 3,
        seed=0,
        observation_horizon=2,
        prediction_horizon=4,
        action_horizon=2,
    )
    dataset = PushTSequenceDataset(
        tiny_dataset,
        episodes=[0],
        normalizer=loaders.normalizer,
        observation_horizon=2,
        prediction_horizon=4,
        action_horizon=2,
    )
    sample = dataset[0]
    observation = loaders.normalizer.observation.denormalize(sample["observation"])
    action = loaders.normalizer.action.denormalize(sample["action"])

    assert torch.equal(observation[:, 0], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(action[:, 0], torch.tensor([0.0, 0.0, 1.0, 2.0]))
    torch.testing.assert_close(
        action[dataset.execution_slice, 0], torch.tensor([0.0, 1.0])
    )

    final_action = loaders.normalizer.action.denormalize(dataset[-1]["action"])
    torch.testing.assert_close(
        final_action[:, 0], torch.tensor([2.0, 3.0, 4.0, 4.0])
    )
    torch.testing.assert_close(
        final_action[dataset.execution_slice, 0], torch.tensor([3.0, 4.0])
    )


def test_split_is_disjoint_and_normalizer_uses_only_train(tiny_dataset: Path) -> None:
    loaders = create_pusht_dataloaders(
        tiny_dataset,
        validation_fraction=1 / 3,
        seed=1,
        observation_horizon=2,
        prediction_horizon=4,
        action_horizon=2,
    )
    assert set(loaders.train_episodes).isdisjoint(loaders.validation_episodes)
    assert sorted(loaders.train_episodes + loaders.validation_episodes) == [0, 1, 2]

    ends = np.array([5, 11, 18])
    starts = np.array([0, 5, 11])
    train_values = np.concatenate(
        [np.arange(starts[i], ends[i]) for i in loaders.train_episodes]
    )
    assert loaders.normalizer.action.minimum[0] == train_values.min()
    assert loaders.normalizer.action.maximum[0] == train_values.max()


def test_batch_shapes_for_state_and_keypoints(tiny_dataset: Path) -> None:
    for observation_type, features in [("state", 5), ("keypoints", 20)]:
        loaders = create_pusht_dataloaders(
            tiny_dataset,
            batch_size=3,
            validation_fraction=1 / 3,
            seed=0,
            observation_type=observation_type,
            observation_horizon=2,
            prediction_horizon=4,
            action_horizon=2,
        )
        batch = next(iter(loaders.train))
        assert batch["observation"].shape == (3, 2, features)
        assert batch["action"].shape == (3, 4, 2)
        assert batch["observation"].dtype == torch.float32
        assert batch["action"].dtype == torch.float32


def test_seed_controls_split_and_batch_order(tiny_dataset: Path) -> None:
    options = dict(
        dataset_path=tiny_dataset,
        batch_size=3,
        validation_fraction=1 / 3,
        seed=7,
        observation_horizon=2,
        prediction_horizon=4,
        action_horizon=2,
    )
    first = create_pusht_dataloaders(**options)
    second = create_pusht_dataloaders(**options)
    assert first.train_episodes == second.train_episodes
    assert first.validation_episodes == second.validation_episodes
    first_batch = next(iter(first.train))
    second_batch = next(iter(second.train))
    torch.testing.assert_close(
        first_batch["observation"], second_batch["observation"]
    )
    torch.testing.assert_close(first_batch["action"], second_batch["action"])

    unseeded = create_pusht_dataloaders(
        tiny_dataset,
        validation_fraction=1 / 3,
        observation_horizon=2,
        prediction_horizon=4,
        action_horizon=2,
    )
    assert unseeded.train.generator is None


def test_normalization_round_trip_for_numpy_and_torch() -> None:
    stats = MinMaxStats(
        minimum=np.array([-2.0, 3.0], dtype=np.float32),
        maximum=np.array([2.0, 3.0], dtype=np.float32),
    )
    values = np.array([[-2.0, 3.0], [0.0, 3.0], [2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(stats.denormalize(stats.normalize(values)), values)
    tensor = torch.from_numpy(values)
    torch.testing.assert_close(stats.denormalize(stats.normalize(tensor)), tensor)
