# Plan for fixing training

## 1. Show two kind of losses separately in wandb

### 1.1. `compute_loss` method & `compute_validation_losses` method

See the `compute_loss` method in `baselines/openpi/src/openpi/models/pi0.py`

There are two kinds of losses:

1. The loss for the diffusion model
2. The loss for the subtask generation

We need to show these two losses separately in wandb.

While returning the total loss, we should also return the two losses separately.

Also, we should synchronize adding the two losses in the return statement in the `compute_validation_losses` method in `baselines/openpi/scripts/train_val.py`

### 1.2. `train_step` method

See the `train_step` method in `baselines/openpi/scripts/train_val.py`

We should add keys of two separate losses in the `info` dictionary so that we can log them separately in wandb.











