export CUDA_VISIBLE_DEVICES=1,2

exp_name="pick_and_place"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_xlerobot.py pi05_xlerobot \
    --exp_name=$exp_name \
    --overwrite \
    --batch_size=32 \
    --num_train_steps=50000 \
    --weight_loader.params_path=gs://openpi-assets/checkpoints/pi05_base/params
