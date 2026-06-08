export CUDA_VISIBLE_DEVICES=2,3,4,5

exp_name="subtask"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_val.py pi05_b1k \
    --exp_name=$exp_name \
    --overwrite \
    --batch_size=64 \
    --num_train_steps=50000 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi05_base/params