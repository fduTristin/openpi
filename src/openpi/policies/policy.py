from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.models import tokenizer as _tokenizer

from PIL import Image

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        high_level_transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._high_level_input_transform_ = _transforms.compose(high_level_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self.jit_sample_low_level_task = nnx_utils.module_jit(self._model.sample_low_level_task, static_argnums=(3,))

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
    
    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if not self._is_pytorch_model:
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            sample_rng_or_pytorch_device = self._pytorch_device

        # two stage inference -- generate subtask & sample actions
        # stage 1: use the dedicated high-level transform pipeline to prepare model inputs.
        high_level_inputs = jax.tree.map(lambda x: x, obs)
        # Placeholder low-level subtask required by TokenizeHighLowPrompt.
        high_level_inputs["subtask"] = np.asarray("ABCDEFG")
        high_level_inputs = self._high_level_input_transform_(high_level_inputs)
        if not self._is_pytorch_model:
            high_level_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], high_level_inputs)
            high_level_rng = sample_rng_or_pytorch_device
        else:
            high_level_inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], high_level_inputs
            )
            high_level_rng = sample_rng_or_pytorch_device

        observation = _model.Observation.from_dict(high_level_inputs)
        observation = _model.preprocess_observation(
            high_level_rng if not self._is_pytorch_model else None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
        )
        
        # Set the low level task tokens to padding according to the loss mask (loss mask is the indication of low-level prompt)
        # We move it from inside model to outside because the inside func need to be jittable
        loss_mask = jnp.array(observation.token_loss_mask)
        new_tokenized_prompt = observation.tokenized_prompt.at[loss_mask].set(0)
        new_tokenized_prompt_mask = observation.tokenized_prompt_mask.at[loss_mask].set(False)
        new_observation = _model.Observation(
                            images=observation.images,
                            image_masks=observation.image_masks,
                            state=observation.state,
                            tokenized_prompt=new_tokenized_prompt,
                            tokenized_prompt_mask=new_tokenized_prompt_mask,
                            token_ar_mask=observation.token_ar_mask,
                            token_loss_mask=observation.token_loss_mask,
                            )
        new_observation = _model.preprocess_observation(None, new_observation, train=False)
        PALIGEMMA_EOS_TOKEN = 1
        max_decoding_steps = 25
        temperature = 0.1
        if not self._is_pytorch_model:
            _predicted_token, _kv_cache, _mask, _ar_mask = self.jit_sample_low_level_task(
                high_level_rng, new_observation, max_decoding_steps, PALIGEMMA_EOS_TOKEN, temperature
            )
            predicted_token_np = np.array(_predicted_token)
            tokenizer = _tokenizer.PaligemmaTokenizer(max_len=max(observation.tokenized_prompt.shape[-1], max_decoding_steps))
            predicted_texts = [
                tokenizer.detokenize(np.asarray(predicted_token_np[i], dtype=np.int32))
                for i in range(predicted_token_np.shape[0])
            ]
            logging.info("[HighLevel] predicted_text=%s", predicted_texts)
            print(f"[HighLevel] predicted_text={predicted_texts}", flush=True)
        else:
            logging.info("[HighLevel] skipped stage-1 decode for PyTorch model.")
            print("[HighLevel] skipped stage-1 decode for PyTorch model.", flush=True)

        # =======================================================================================
        # stage 2: sample actions, temporarily irrelevant of subtask generation 
        inputs = jax.tree.map(lambda x: x, obs)
        
        # extract image from inputs and save for debugging
        if 'observation/egocentric_camera' in inputs:
            img = inputs['observation/egocentric_camera']
            # save image for debugging
            if isinstance(img, jnp.ndarray):
                img = np.array(img)
            img = Image.fromarray(img.astype('uint8'))
            img.save('base_0_rgb.png')
        
        # if 'observation/wrist_image_left' in inputs:
        #     img = inputs['observation/wrist_image_left']
        #     # save image for debugging
        #     if isinstance(img, jnp.ndarray):
        #         img = np.array(img)
        #     img = Image.fromarray(img.astype('uint8'))
        #     img.save('left_wrist_0_rgb.png')
        
        # if 'observation/wrist_image_right' in inputs:
        #     img = inputs['observation/wrist_image_right']
        #     # save image for debugging
        #     if isinstance(img, jnp.ndarray):
        #         img = np.array(img)
        #     img = Image.fromarray(img.astype('uint8'))
        #     img.save('right_wrist_0_rgb.png')
        
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
