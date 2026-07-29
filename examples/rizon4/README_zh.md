# Rizon4 数据集转换

本示例会将 `../data/episodes/task_1` 中的本地数据集转换为 `openpi` 可直接使用的 LeRobot 数据集。

## 转换器做了什么

原始数据集的组织结构如下：

```text
data/episodes/task_1/
  episode_0000/
    metadata.json
    actions.joint_position/data.csv
    observation.state.joint_position/data.csv
    videos/*.mp4
    videos/*_timestamps.csv
```

转换器会写出一个 DROID 风格的 LeRobot 数据集，包含以下字段：

- `observation.exterior_image_1_left`
- `observation.exterior_image_2_left`
- `observation.wrist_image_left`
- `observation.tactile_image`
- `observation.joint_position`
- `observation.gripper_position`
- `actions`
- `task`

默认情况下，脚本会按时间戳对齐所有数据流，并使用以下相机：

- `cam_0915` 作为第一个第三视角相机
- `cam_2595` 作为第二个第三视角相机
- `cam_2546` 作为腕部相机
- `cam_usbv2-0_7.4` 作为触觉相机

## 运行转换

从 `openpi` 仓库根目录执行：

```bash
uv run examples/rizon4/convert_rizon4_data_to_lerobot.py \
  --data-dir /home/xsuper/WorkSpace/xhz/data/episodes \
  --repo-id your_hf_username/rizon4_task1
```

如果想先做一个快速验证，可以只转换少量 episode：

```bash
uv run examples/rizon4/convert_rizon4_data_to_lerobot.py \
  --data-dir /home/xsuper/WorkSpace/xhz/data/episodes \
  --repo-id your_hf_username/rizon4_task1 \
  --max-episodes 2
```

如果你的相机命名和默认值不同，可以直接覆盖：

```bash
uv run examples/rizon4/convert_rizon4_data_to_lerobot.py \
  --data-dir /home/xsuper/WorkSpace/xhz/data/episodes \
  --repo-id your_hf_username/rizon4_task1 \
  --external-camera cam_0915 \
  --secondary-external-camera cam_2595 \
  --wrist-camera cam_2546 \
  --tactile-camera cam_usbv2-0_7.4
```

## 使用 openpi 训练

转换后的数据集可以直接作为 LeRobot 数据集使用。先为 Flexiv 配置计算归一化统计：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_flexiv
```

然后使用仓库里不带触觉的 Flexiv 训练配置：

```bash
uv run scripts/train.py pi05_flexiv --exp_name my_flexiv_run --overwrite
```

如果想用 pi0-FAST，可以运行：

```bash
uv run scripts/train.py pi0_fast_flexiv_finetune --exp_name my_flexiv_run --overwrite
```

这两个配置只使用非触觉字段，并将 `actions` 按“7 个关节增量 + 绝对 gripper”来训练。`pi05_flexiv` 默认从 `gs://openpi-assets/checkpoints/pi05_base/params` 初始化模型，归一化统计会写入你本地的 checkpoint 资产目录。

## 在线部署到 Rizon4

在模型机器上用训练好的 checkpoint 启动 websocket policy server：

```bash
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_flexiv \
  --policy.dir=checkpoints/pi05_flexiv/my_flexiv_run/30000
```

在连接 Rizon4 和相机的 client 机器上先检查相机名：

```bash
uv run examples/rizon4/flexiv_client.py --list-cameras
```

确认无误后启动 client。模型输出的前 7 维会按关节增量累加后发送给机械臂，第 8 维作为归一化 gripper 绝对开合位置执行：

```bash
uv run examples/rizon4/flexiv_client.py \
  --host <server_ip> \
  --port 8000 \
  --instruction "task 1"
```

如果现场相机名不同，可以覆盖映射：

```bash
uv run examples/rizon4/flexiv_client.py \
  --host <server_ip> \
  --camera-map exterior_image_1_left=cam_0915 \
  --camera-map exterior_image_2_left=cam_2595 \
  --camera-map wrist_image_left=cam_2546
```

## 查看输出

数据集会写入 `openpi` 使用的本地 LeRobot 缓存目录（见 LeRobot 包中的 `LEROBOT_HOME` / `HF_LEROBOT_HOME`）。

在默认 `use_videos=True` 的情况下，相机流会以 `.mp4` 形式保存在该缓存目录中。如果想改成图片形式，可以运行时传 `--use-videos False`。
