from openpi_client import action_chunk_broker
import pytest
import numpy as np

from openpi.policies import aloha_policy
from openpi.policies import flexiv_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


def test_flexiv_inputs():
    data = flexiv_policy.make_flexiv_example()
    data["actions"] = np.ones((10, 8))
    inputs = flexiv_policy.FlexivInputs(model_type=_config.get_config("pi0_flexiv_finetune").model.model_type)(data)

    assert set(inputs["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert inputs["state"].shape == (8,)
    assert inputs["actions"].shape == (10, 8)


def test_flexiv_outputs():
    outputs = flexiv_policy.FlexivOutputs()({"actions": np.ones((10, 12))})
    assert outputs["actions"].shape == (10, 8)
