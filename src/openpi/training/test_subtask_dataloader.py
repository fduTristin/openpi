"""
Test the full transform pipeline for BehaviorLeRobotDataset -> stage1 subtask.
Verifies that subtask survives the full pipeline:
  Dataset -> PromptFromLeRobotTask -> RepackTransform -> B1kInputs -> Normalize -> TokenizeHighLowPrompt
"""

import sys
sys.path.insert(0, "/home/xhz/b1k-baselines/baselines/openpi/src")

import numpy as np
from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset
from openpi.training import config as _config
import openpi.training.sharding as sharding
import jax
import logging
from openpi.training import utils as training_utils
from openpi.training import data_loader as _dataloader

def test_subtask_pipeline(config: _config.TrainConfig):
    # Create the dataloader
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    dataloader = _dataloader.create_behavior_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        skip_norm_stats=False,
    )
    data_iter = iter(dataloader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    for i in range(10):
        batch = next(data_iter)
        logging.info(f"Batch {i}:\n{training_utils.array_tree_to_info(batch)}")

if __name__ == "__main__":
    # set log path to "./logs/test_subtask_dataloader.log"
    import logging
    import atexit

    log_path = "./logs/test_subtask_dataloader.log"

    # Clear existing handlers to avoid duplicate/color-coded logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # Suppress noisy loggers from dependencies
    for noisy in ["matplotlib", "PIL", "huggingface_hub", "omnigibson"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    test_subtask_pipeline(_config.cli())
