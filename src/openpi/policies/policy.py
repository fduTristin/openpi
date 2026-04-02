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
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        
        # logging.info(f"Policy input keys: {list(inputs.keys())}")
        # logging.info(f"Policy prompt: {inputs.get('prompt', None)}")
        
        img_dict = {
            'base_0_rgb': None,
            'left_wrist_0_rgb': None,
            'right_wrist_0_rgb': None,
        }
        # extract image from inputs and save for debugging
        if 'observation/egocentric_camera' in inputs:
            img = inputs['observation/egocentric_camera']
            img_dict['base_0_rgb'] = jnp.array(img[np.newaxis, :, :, :]).astype(jnp.float32) / 127.5 - 1.0
            # save image for debugging
            if isinstance(img, jnp.ndarray):
                img = np.array(img)
            img = Image.fromarray(img.astype('uint8'))
            img.save('base_0_rgb.png')
        
        if 'observation/wrist_image_left' in inputs:
            img = inputs['observation/wrist_image_left']
            img_dict['left_wrist_0_rgb'] = jnp.array(img[np.newaxis, :, :, :]).astype(jnp.float32) / 127.5 - 1.0
            # save image for debugging
            if isinstance(img, jnp.ndarray):
                img = np.array(img)
            img = Image.fromarray(img.astype('uint8'))
            img.save('left_wrist_0_rgb.png')
        
        if 'observation/wrist_image_right' in inputs:
            img = inputs['observation/wrist_image_right']
            img_dict['right_wrist_0_rgb'] = jnp.array(img[np.newaxis, :, :, :]).astype(jnp.float32) / 127.5 - 1.0
            # save image for debugging
            if isinstance(img, jnp.ndarray):
                img = np.array(img)
            img = Image.fromarray(img.astype('uint8'))
            img.save('right_wrist_0_rgb.png')
        
        # extract high-level prompt before input transformation
        high_level_prompt = inputs.get('prompt','')
        
        inputs = self._input_transform(inputs)

        # two stage inference -- generate subtask & sample actions
        
        # stage 1: generate subtask from high-level prompt
        print('======================', flush=True)
        print(f"\033[32m[HIGH LEVEL PROMPT]\033[0m {high_level_prompt}", flush=True)
        low_level_prompt = 'ABCDEFG' # placeholder for low-level prompt extraction logic
        tokenizer = _tokenizer.PaligemmaTokenizer(max_len=128)
        tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = tokenizer.tokenize_high_low_prompt(high_level_prompt, low_level_prompt)
        data = {
            'image': img_dict,
            'image_mask': {key: jnp.ones(1, dtype=jnp.bool) for key in img_dict.keys()},
            'state': jnp.zeros((1, 32), dtype=jnp.float32),
            'tokenized_prompt': jnp.stack([tokenized_prompt], axis=0),
            'tokenized_prompt_mask': jnp.stack([tokenized_prompt_mask], axis=0),
            'token_ar_mask': jnp.stack([token_ar_mask], axis=0),
            'token_loss_mask': jnp.stack([token_loss_mask], axis=0),
        }
        observation = _model.Observation.from_dict(data)
        rng = jax.random.key(42)
        observation = _model.preprocess_observation(rng, observation, train=False, image_keys=list(observation.images.keys()))
        
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
        observation = _model.preprocess_observation(None, new_observation, train=False, image_keys=list(observation.images.keys()))
        observation = jax.tree.map(jax.device_put, observation)
        PALIGEMMA_EOS_TOKEN = 1
        max_decoding_steps = 25
        temperature = 0.1
        predicted_token, kv_cache, mask, ar_mask = self.jit_sample_low_level_task(rng, observation, max_decoding_steps, PALIGEMMA_EOS_TOKEN, temperature)
        for i in range(predicted_token.shape[0]):
            print(f"\033[31m[PRED]\033[0m " + tokenizer.detokenize(np.array(predicted_token[i], dtype=np.int32)), flush=True)
            print(f"\033[31m[MASK]\033[0m " + tokenizer.detokenize(np.array(data['tokenized_prompt'], dtype=np.int32)), flush=True)
        
        print('======================', flush=True)
        
        # stage 2: sample actions, temporarily irrelevant of subtask generation
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

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
