# Rizon4 Dataset Conversion

This example converts the local dataset in `../data/episodes/task_1` into a
LeRobot dataset that can be consumed by `openpi`.

## What the converter does

The raw dataset is organized as:

```text
data/episodes/task_1/
  episode_0000/
    metadata.json
    actions.joint_position/data.csv
    observation.state.joint_position/data.csv
    videos/*.mp4
    videos/*_timestamps.csv
```

The converter writes a DROID-style LeRobot dataset with these fields:

- `observation.exterior_image_1_left`
- `observation.exterior_image_2_left`
- `observation.wrist_image_left`
- `observation.tactile_image`
- `observation.joint_position`
- `observation.gripper_position`
- `actions`
- `task`

By default, the script aligns all streams by timestamp, using:

- `cam_0915` as the first third-view camera
- `cam_2595` as the second third-view camera
- `cam_2546` as the wrist camera
- `cam_usbv2-0_7.4` as the tactile camera

## Run conversion

From the `openpi` repository root:

```bash
uv run examples/rizon4/convert_rizon4_data_to_lerobot.py \
  --data-dir /home/xsuper/WorkSpace/xhz/data/episodes \
  --repo-id your_hf_username/rizon4_task1
```

For a quick smoke test, convert only a few episodes first:

```bash
uv run examples/rizon4/convert_rizon4_data_to_lerobot.py \
  --data-dir /home/xsuper/WorkSpace/xhz/data/episodes \
  --repo-id your_hf_username/rizon4_task1 \
  --max-episodes 2
```

If your cameras differ from the defaults, override them directly:

```bash
uv run examples/rizon4/convert_rizon4_data_to_lerobot.py \
  --data-dir /home/xsuper/WorkSpace/xhz/data/episodes \
  --repo-id your_hf_username/rizon4_task1 \
  --external-camera cam_0915 \
  --secondary-external-camera cam_2595 \
  --wrist-camera cam_2546 \
  --tactile-camera cam_usbv2-0_7.4
```

## Train with openpi

The converted dataset is ready to use as a LeRobot dataset. To train with
`openpi`, first compute normalization stats for the Flexiv config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_flexiv
```

Then start training with the non-tactile Flexiv config:

```bash
uv run scripts/train.py pi05_flexiv --exp_name my_flexiv_run --overwrite
```

If you prefer pi0-FAST, use:

```bash
uv run scripts/train.py pi0_fast_flexiv_finetune --exp_name my_flexiv_run --overwrite
```

These configs use the non-tactile dataset fields only and treat `actions` as
7 joint deltas plus an absolute gripper value. `pi05_flexiv` initializes from
`gs://openpi-assets/checkpoints/pi05_base/params`, and the stats computed above
will be stored in your local checkpoint assets.

## Deploy Online To Rizon4

On the model machine, start the websocket policy server from a trained checkpoint:

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_flexiv \
  --policy.dir=checkpoints/pi05_flexiv/my_flexiv_run/30000
```

On the client machine connected to the Rizon4 and cameras, first inspect camera names:

```bash
uv run examples/rizon4/flexiv_client.py --list-cameras
```

After verifying the camera map, start the client. The first 7 model output
dimensions are accumulated as joint deltas before being sent to the robot; the
8th dimension is executed as an absolute normalized gripper position:

```bash
uv run examples/rizon4/flexiv_client.py \
  --host <server_ip> \
  --port 8000 \
  --instruction "task 1"
```

If the on-site camera names differ, override the mapping:

```bash
uv run examples/rizon4/flexiv_client.py \
  --host <server_ip> \
  --camera-map exterior_image_1_left=cam_0915 \
  --camera-map exterior_image_2_left=cam_2595 \
  --camera-map wrist_image_left=cam_2546
```

## Inspecting the output

The dataset will be written to the local LeRobot cache directory used by `openpi`
(see `LEROBOT_HOME` / `HF_LEROBOT_HOME` in the LeRobot package).

With `use_videos=True` (the default), the camera streams are stored as `.mp4`
files under that cache directory. If you want image files instead, run with
`--use-videos False`.
