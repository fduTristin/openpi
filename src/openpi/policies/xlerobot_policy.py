"""Policy transforms for XLeRobot datasets.

Provides input/output transforms that convert XLeRobot dataset observations into
the format expected by pi0/pi0.5 models, and convert model outputs back to
actions suitable for the XLeRobot robot.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 HWC format.

    LeRobot stores images as float32 (C,H,W); this converts to uint8 (H,W,C).
    Already-parsed images are passed through unchanged.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class XLeRobotInputs(transforms.DataTransformFn):
    """Transform XLeRobot observations into model inputs.

    Expects the following keys in the input dict (after repack):
        - observation.state:       (17,) raw joint/state vector
        - observation.images.head:  head camera image (H,W,C,3) or (3,H,W)
        - observation.images.left_wrist:  left wrist image
        - observation.images.right_wrist:  right wrist image

    Produces:
        - state:        (23,) joint state (arm joints + base + gripper)
        - image:        dict of camera-name -> uint8 HWC image
        - image_mask:   dict of camera-name -> True (all cameras present)
        - actions:      action chunk (passthrough)
    """

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # list data keys
        # print(data.keys())
        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape[-1] != self.action_dim:
            state = np.pad(state, ((0, self.action_dim - state.shape[-1])), mode="constant")

        base_image = _parse_image(data["observation.images.head"])
        wrist_image_left = _parse_image(data["observation.images.left_wrist"])
        wrist_image_right = _parse_image(data["observation.images.right_wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, wrist_image_left, wrist_image_right)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "action" in data:
            inputs["actions"] = data["action"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XLeRobotOutputs(transforms.DataTransformFn):
    """Transform model outputs back to XLeRobot action format.

    Takes raw model action output and slices/pads to the XLeRobot action dimension.
    """

    action_dim: int = 17

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] > self.action_dim:
            actions = actions[..., : self.action_dim]
        elif actions.shape[-1] < self.action_dim:
            actions = np.pad(actions, ((0, self.action_dim - actions.shape[-1])), mode="constant")
        return {"actions": actions}


def make_xlerobot_example() -> dict:
    """Creates a random input example for the XLeRobot policy."""
    return {
        "observation.state": np.random.rand(17).astype(np.float32),
        "observation.images.head": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation.images.right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick and place the object",
    }
