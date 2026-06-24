"""Test the XLeRobot dataset and data loader pipeline.

Verifies that XLeRobotDataset loads correctly and the data loader produces batches
with the expected structure and shapes.
"""

import dataclasses
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import jax
from openpi.training import sharding
from openpi.training import data_loader as _dataloader
from openpi.training import xlerobot_dataset as _xlerobot_dataset


def test_xlerobot_dataset_basic():
    """Test that XLeRobotDataset loads and returns the expected keys."""
    root = os.path.expanduser("~/Datasets/xlerobot/pick_and_place")
    if not os.path.exists(root):
        print(f"Skipping: dataset not found at {root}")
        return

    dataset = _xlerobot_dataset.XLeRobotDataset(
        root=root,
        episodes=list(range(3)),
    )

    assert len(dataset) > 0, "Dataset should not be empty"
    sample = dataset[0]

    # Check required keys
    assert "observation/state" in sample, "Missing observation/state"
    assert "actions" in sample, "Missing actions"
    assert "task_index" in sample, "Missing task_index"
    assert "episode_index" in sample, "Missing episode_index"
    assert "frame_index" in sample, "Missing frame_index"

    # Check shapes
    assert sample["observation/state"].shape == (17,), f"State shape mismatch: {sample['observation/state'].shape}"
    assert sample["actions"].shape == (1, 17), f"Action shape mismatch: {sample['actions'].shape} (expected (action_horizon, action_dim))"

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Tasks: {dataset.tasks}")
    print(f"Meta: robot_type={dataset.meta.get('robot_type')}, fps={dataset.meta.get('fps')}, total_episodes={dataset.meta.get('total_episodes')}")


def test_xlerobot_dataset_episode_filtering():
    """Test that episode filtering works correctly."""
    root = os.path.expanduser("~/Datasets/xlerobot/pick_and_place")
    if os.path.exists(root):
        dataset_all = _xlerobot_dataset.XLeRobotDataset(root=root, episodes=None)
        dataset_filtered = _xlerobot_dataset.XLeRobotDataset(root=root, episodes=[0, 1])
        assert len(dataset_filtered) < len(dataset_all), "Filtered dataset should have fewer samples"
        # Verify all returned samples are from episodes 0 or 1
        for i in range(min(5, len(dataset_filtered))):
            ep = dataset_filtered[i]["episode_index"]
            assert ep in [0, 1], f"Sample {i} has episode {ep}, expected 0 or 1"
    else:
        print(f"Skipping: dataset not found at {root}")

def test_create_xlerobot_data_loader_with_train_config():
    """Test create_xlerobot_data_loader using the pi05_xlerobot TrainConfig."""
    from openpi.training import config as _cfg

    train_config = _cfg.get_config("pi05_xlerobot")

    # Resolve to the local dataset path. The config's xlerobot_dataset_root may point to a
    # different machine's mount (e.g. /mnt/sdb/xhz/Datasets/xlerobot). The dataset root
    # is the PARENT of the "pick_and_place" subdirectory, since create_xlerobot_dataset
    # appends repo_id="pick_and_place" to the root path.
    local_root = os.path.expanduser("~/Datasets/xlerobot")
    remote_root = os.path.expanduser(train_config.data.base_config.xlerobot_dataset_root)
    # Prefer the local path if the dataset exists there; otherwise use the config path.
    if os.path.exists(os.path.join(local_root, "pick_and_place", "meta", "info.json")):
        root = local_root
    elif os.path.exists(os.path.join(remote_root, "pick_and_place", "meta", "info.json")):
        root = remote_root
    else:
        root = None
    if root is not None:
        base_cfg = dataclasses.replace(train_config.data.base_config, xlerobot_dataset_root=root)
        train_config = dataclasses.replace(train_config, data=dataclasses.replace(train_config.data, base_config=base_cfg))
        print(f"Using dataset root: {os.path.join(root, 'pick_and_place')}")
    else:
        print(f"Skipping: dataset not found (checked {root})")
        return

    rng = jax.random.key(train_config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    # Create the data loader with pytorch framework and a small num_batches for testing
    data_loader = _dataloader.create_xlerobot_data_loader(
        train_config,
        sharding=data_sharding,
        shuffle=True,
        num_batches=5
    )

    data_iter = iter(data_loader)
    for i in range(5):
        print(f"Batch {i}:")
        obs, actions = next(data_iter)
        # Actions: (batch_size, action_horizon, action_dim)
        assert actions.shape[0] == train_config.batch_size, (
            f"Batch size should be {train_config.batch_size}, got {actions.shape[0]}"
        )
        assert actions.shape[1] == train_config.model.action_horizon, (
            f"Action horizon should be {train_config.model.action_horizon}, got {actions.shape[1]}"
        )
        assert actions.shape[2] == train_config.model.action_dim, (
            f"Action dim should be {train_config.model.action_dim}, got {actions.shape[2]}"
        )
        # Verify Observation fields are present
        assert obs.images is not None, "Observation.images should not be None"
        assert obs.image_masks is not None, "Observation.image_masks should not be None"
        assert obs.state is not None, "Observation.state should not be None"
        assert obs.state.shape == (train_config.batch_size, train_config.model.action_dim), (
            f"State shape should be ({train_config.batch_size}, {train_config.model.action_dim}), "
            f"got {obs.state.shape}"
        )
        logging.info(
            f"Batch: actions shape={actions.shape}, "
            f"state shape={obs.state.shape}, image keys={list(obs.images.keys()) if obs.images else None}"
        )

    logging.info("create_xlerobot_data_loader test passed!")


if __name__ == "__main__":
    log_path = "./logs/xlerobot_data_loader_test.log"
    os.makedirs("./logs", exist_ok=True)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    for noisy in ["matplotlib", "PIL", "huggingface_hub", "jax"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print("=" * 60)
    print("test_xlerobot_dataset_basic")
    print("=" * 60)
    test_xlerobot_dataset_basic()

    print("=" * 60)
    print("test_xlerobot_dataset_episode_filtering")
    print("=" * 60)
    test_xlerobot_dataset_episode_filtering()

    print("=" * 60)
    print("test_create_xlerobot_data_loader_with_train_config")
    print("=" * 60)
    test_create_xlerobot_data_loader_with_train_config()

    print("\nAll tests passed!")
