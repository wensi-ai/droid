"""Live monocular depth estimation backed by Video Depth Anything.

The upstream streaming model is stateful, so this wrapper keeps an independent
model instance for each camera stream. Imports are intentionally lazy so normal
DROID operation does not require PyTorch unless depth inference is enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class VideoDepthAnythingEstimator:
    """Estimate temporally consistent relative depth for multiple RGB streams."""

    def __init__(
        self,
        *,
        model_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        encoder: str = "vits",
        input_size: int = 518,
        device: str | None = None,
        fp32: bool = False,
        normalization_percentiles: tuple[float, float] = (2.0, 98.0),
        normalization_momentum: float = 0.1,
    ):
        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported encoder {encoder!r}; choose from {sorted(MODEL_CONFIGS)}")
        if input_size <= 0:
            raise ValueError("input_size must be positive")

        low_percentile, high_percentile = normalization_percentiles
        if not 0 <= low_percentile < high_percentile <= 100:
            raise ValueError("normalization_percentiles must satisfy 0 <= low < high <= 100")
        if not 0 < normalization_momentum <= 1:
            raise ValueError("normalization_momentum must be in (0, 1]")

        default_root = Path(__file__).resolve().parents[3] / "Video-Depth-Anything"
        self.model_root = Path(model_root).expanduser().resolve() if model_root else default_root
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path
            else self.model_root / "checkpoints" / f"video_depth_anything_{encoder}.pth"
        )
        if not self.model_root.is_dir():
            raise FileNotFoundError(
                f"Video Depth Anything checkout not found at {self.model_root}. "
                "Set --depth-model-root to its checkout directory."
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Video Depth Anything checkpoint not found at {self.checkpoint_path}. "
                "Download the matching relative-depth checkpoint first."
            )

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Depth inference requires PyTorch in the active environment. "
                "Install the Video Depth Anything dependencies before using --depth."
            ) from exc

        model_root_str = str(self.model_root)
        if model_root_str not in sys.path:
            sys.path.insert(0, model_root_str)
        try:
            from video_depth_anything.video_depth_stream import VideoDepthAnything
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Could not import Video Depth Anything from {self.model_root}. "
                "Install its dependencies in the active environment."
            ) from exc

        self._torch = torch
        self._model_class = VideoDepthAnything
        self.encoder = encoder
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp32 = fp32 or self.device == "cpu"
        self.normalization_percentiles = normalization_percentiles
        self.normalization_momentum = normalization_momentum
        self._models = {}
        self._normalization_ranges = {}

        # Load the checkpoint once on CPU, then reuse it for each stream model.
        self._state_dict = torch.load(self.checkpoint_path, map_location="cpu")

    def _make_model(self):
        model = self._model_class(**MODEL_CONFIGS[self.encoder])
        model.load_state_dict(self._state_dict, strict=True)
        return model.to(self.device).eval()

    def _get_model(self, stream_name: str):
        if stream_name not in self._models:
            self._models[stream_name] = self._make_model()
        return self._models[stream_name]

    def infer(self, rgb_frame: np.ndarray, *, stream_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(depth_image, raw_depth)`` for one RGB frame.

        ``depth_image`` is an HWC, three-channel uint8 visualization suitable
        for image-only policies. ``raw_depth`` is the model's float depth map.
        """
        rgb_frame = np.asarray(rgb_frame)
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 RGB frame, got shape {rgb_frame.shape}")
        if rgb_frame.dtype != np.uint8:
            rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)

        model = self._get_model(stream_name)
        raw_depth = model.infer_video_depth_one(
            rgb_frame,
            input_size=self.input_size,
            device=self.device,
            fp32=self.fp32,
        )
        depth_image = self.depth_to_image(raw_depth, stream_name=stream_name)
        return depth_image, raw_depth

    def depth_to_image(self, depth: np.ndarray, *, stream_name: str) -> np.ndarray:
        """Robustly normalize a relative-depth map into three grayscale channels."""
        depth = np.asarray(depth, dtype=np.float32)
        finite = np.isfinite(depth)
        if not finite.any():
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        low_percentile, high_percentile = self.normalization_percentiles
        current_low, current_high = np.percentile(depth[finite], [low_percentile, high_percentile])
        if stream_name in self._normalization_ranges:
            previous_low, previous_high = self._normalization_ranges[stream_name]
            momentum = self.normalization_momentum
            current_low = (1 - momentum) * previous_low + momentum * current_low
            current_high = (1 - momentum) * previous_high + momentum * current_high
        self._normalization_ranges[stream_name] = (current_low, current_high)

        scale = max(float(current_high - current_low), np.finfo(np.float32).eps)
        normalized = np.clip((depth - current_low) / scale, 0.0, 1.0)
        normalized[~finite] = 0.0
        grayscale = np.rint(normalized * 255).astype(np.uint8)
        return np.repeat(grayscale[..., None], 3, axis=2)

    def reset(self):
        """Clear temporal and visualization state while retaining loaded weights."""
        for model in self._models.values():
            model.transform = None
            model.frame_id_list = []
            model.frame_cache_list = []
            model.id = -1
            for attribute in ("frame_height", "frame_width"):
                if hasattr(model, attribute):
                    delattr(model, attribute)
        self._normalization_ranges.clear()

