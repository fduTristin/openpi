"""Dataset loader for XLeRobot datasets in LeRobot format."""

from __future__ import annotations

import glob
import importlib
import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import get_worker_info


# ---------------------------------------------------------------------------
# Video utilities (based on lerobot.datasets.video_utils)
# ---------------------------------------------------------------------------

_TORCHCODEC_AVAILABLE = importlib.util.find_spec("torchcodec") is not None


def _get_video_path(root: str, video_key: str, chunk_index: int, file_index: int) -> str:
    """Get the absolute path to the MP4 file for a given chunk_index and file_index."""
    return os.path.join(
        root,
        "videos",
        video_key,
        f"chunk-{chunk_index:03d}",
        f"file-{file_index:03d}.mp4",
    )


class _VideoCache:
    """Per-worker video cache using torchcodec.

    Uses torchcodec's get_frames_at() API which is 65x faster than pyav's
    VideoReader for random access (common with DataLoader shuffle).
    """

    def __init__(self):
        self._decoders: dict[str, Any] = {}
        self._ep_idx: int | None = None

    def get_frame(
        self,
        root: str,
        video_key: str,
        chunk_index: int,
        file_index: int,
        frame_index: int,
        ep_idx: int,
    ) -> np.ndarray:
        """Return a single uint8 HWC RGB frame."""
        if not _TORCHCODEC_AVAILABLE:
            raise ImportError(
                "torchcodec is required for video decoding. Install with: pip install torchcodec"
            )

        video_path = _get_video_path(root, video_key, chunk_index, file_index)

        if not os.path.exists(video_path):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Episode changed — close old decoders
        if self._ep_idx != ep_idx:
            self._close_decoders()
            self._ep_idx = ep_idx

        if video_path not in self._decoders:
            from torchcodec.decoders import VideoDecoder

            self._decoders[video_path] = VideoDecoder(video_path, seek_mode="approximate")
        decoder = self._decoders[video_path]

        # Use torchcodec batch API — 65x faster than pyav for random access
        frames_batch = decoder.get_frames_at(indices=[frame_index])
        frame = frames_batch.data[0]
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).numpy()
        else:
            frame = np.asarray(frame)
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
        return frame

    def _close_decoders(self):
        """Close all cached decoders."""
        self._decoders.clear()

    def reset(self):
        """Called when the dataset iterator restarts (epoch boundary)."""
        self._ep_idx = None
        self._close_decoders()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Maps camera name used by the policy to the LeRobot video key.
_VIDEO_KEYS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


class XLeRobotDataset:
    """XLeRobot dataset compatible with LeRobot's data format.

    Loads data from local parquet files organized in LeRobot's chunk format,
    and reads video frames using torchcodec.

    Each sample contains:
        - observation/state: (17,) float32, robot joint state
        - observation/image_head:       (480, 640, 3) uint8 RGB
        - observation/image_left_wrist: (480, 640, 3) uint8 RGB
        - observation/image_right_wrist: (480, 640, 3) uint8 RGB
        - action: (17,) float32, target action
        - task_index: task identifier
    """

    def __init__(
        self,
        root: str,
        *,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
    ):
        """Initialize the XLeRobot dataset.

        Args:
            root: Root directory of the XLeRobot dataset (contains data/ and meta/).
            episodes: Optional list of episode indices to load. If None, all episodes are used.
            delta_timestamps: Optional mapping from key to list of timestamps offset relative
                to the current frame. Not used for XLeRobot since we don't load videos here.
        """
        self._root = root
        self._episodes = episodes
        self._delta_timestamps = delta_timestamps

        meta_dir = os.path.join(root, "meta")
        with open(os.path.join(meta_dir, "info.json"), "r") as f:
            self._meta = json.load(f)

        tasks_path = os.path.join(meta_dir, "tasks.parquet")
        if os.path.exists(tasks_path):
            tasks_df = pd.read_parquet(tasks_path)
            # Row index is the task name, 'task_index' column is the integer index
            self._tasks: dict[int, str] = {int(row["task_index"]): name for name, row in tasks_df.iterrows()}
        else:
            self._tasks = {}

        # Maps file_index → cumulative row count *before* that file.
        # This is used to compute the video frame offset: video_frame = index - offset[file_index].
        self._file_index_offsets: dict[int, int] = {}

        self._build_index()

        # Per-worker video cache. Lazily initialized in __getitem__ to be fork-safe.
        self._video_cache: _VideoCache | None = None

    def _build_index(self) -> None:
        """Build a single sorted DataFrame of all selected samples, indexed by global position."""
        meta_dir = os.path.join(self._root, "meta")
        meta_files = sorted(glob.glob(os.path.join(meta_dir, "episodes", "chunk-*", "*.parquet")))

        # Load episode metadata to get per-episode video (chunk_index, file_index).
        meta_rows: list[pd.DataFrame] = []
        for meta_path in meta_files:
            df = pd.read_parquet(meta_path)
            if self._episodes is not None:
                df = df[df["episode_index"].isin(self._episodes)]
            meta_rows.append(df)
        meta_df = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()

        # Build lookup: episode_index → video (chunk_index, file_index) for each camera.
        self._episode_to_videos: dict[int, dict[str, tuple[int, int]]] = {}
        for _, row in meta_df.iterrows():
            ep_idx = int(row["episode_index"])
            self._episode_to_videos[ep_idx] = {}
            for cam, video_key in _VIDEO_KEYS.items():
                ci = int(row[f"videos/{video_key}/chunk_index"])
                fi = int(row[f"videos/{video_key}/file_index"])
                self._episode_to_videos[ep_idx][video_key] = (ci, fi)

        data_dir = os.path.join(self._root, "data")
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "chunk-*", "*.parquet")))

        all_rows: list[pd.DataFrame] = []
        cumulative_rows = 0
        for pq_path in parquet_files:
            pq_name = os.path.basename(pq_path)
            file_index = int(re.search(r"file-(\d+)\.parquet", pq_name).group(1))

            df = pd.read_parquet(pq_path)
            if self._episodes is not None:
                df = df[df["episode_index"].isin(self._episodes)]
            df["_file_index"] = file_index
            all_rows.append(df)

            self._file_index_offsets[file_index] = cumulative_rows
            cumulative_rows += len(df)

        self._df = pd.concat(all_rows, ignore_index=True)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Lazily create per-worker video cache (must be done inside __getitem__ to be
        # fork-safe in DataLoader workers).
        if self._video_cache is None:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            self._video_cache = _VideoCache()

        row = self._df.iloc[index]

        sample: dict[str, Any] = {
            "observation/state": np.asarray(row["observation.state"], dtype=np.float32),
            "task_index": int(row["task_index"]),
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
        }

        # Build action chunk: look ahead action_horizon frames within the same episode
        action_horizon = 1
        if self._delta_timestamps:
            action_horizon = len(next(iter(self._delta_timestamps.values())))

        ep_idx = int(row["episode_index"])
        frame_idx = int(row["frame_index"])

        action_chunk: list[np.ndarray] = []
        for offset in range(action_horizon):
            look_idx = index + offset
            if look_idx >= len(self._df):
                action_chunk.append(action_chunk[-1])
            else:
                next_row = self._df.iloc[look_idx]
                if int(next_row["episode_index"]) == ep_idx and int(next_row["frame_index"]) == frame_idx + offset:
                    action_chunk.append(np.asarray(next_row["action"], dtype=np.float32))
                else:
                    action_chunk.append(action_chunk[-1])

        sample["actions"] = np.stack(action_chunk, axis=0)

        # Load images from MP4 files using torchcodec.
        # NOTE: The `index` column in parquet is a globally unique, monotonically
        # increasing row number that maps directly to the video frame offset within
        # each file.  frame_index is episode-local and cannot be used here.
        # See https://github.com/huggingface/lerobot/pull/286 for details.
        file_index = int(row["_file_index"])
        video_frame_offset = int(row["index"]) - self._file_index_offsets[file_index]

        video_info = self._episode_to_videos.get(ep_idx, {})
        for cam, video_key in _VIDEO_KEYS.items():
            video_chunk, video_file = video_info.get(video_key, (0, 0))
            frame = self._video_cache.get_frame(
                root=self._root,
                video_key=video_key,
                chunk_index=video_chunk,
                file_index=video_file,
                frame_index=video_frame_offset,
                ep_idx=ep_idx,
            )
            sample[f"observation/image_{cam}"] = frame

        # DEBUG: save image_head to disk
        import cv2
        os.makedirs("debug", exist_ok=True)
        # torchcodec returns RGB, cv2.imwrite expects BGR
        rgb = sample["observation/image_head"]
        bgr = rgb[:, :, ::-1]
        cv2.imwrite(f"debug/image_head_{ep_idx}_{frame_idx}.png", bgr)
        return sample

    @property
    def meta(self) -> dict[str, Any]:
        """Return dataset metadata (robot type, fps, total episodes, etc.)."""
        return self._meta

    @property
    def tasks(self) -> dict[int, str]:
        """Return mapping from task index to task name."""
        return self._tasks
