"""FeatherTalk adapter for prepared, person-specific avatar datasets.

Avatar manifest metadata ``feathertalk`` accepts ``dataset``, ``checkpoint``
and ``hubert_checkpoint`` paths, relative to the avatar bundle. Environment
variables OPENTALKING_FEATHERTALK_{CHECKPOINT,HUBERT_CHECKPOINT} override weights.
Datasets contain full_body_img/<index>.jpg and landmarks/<index>.lms.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from opentalking.avatar.loader import load_avatar_bundle
from opentalking.core.interfaces.avatar_asset import AvatarManifest
from opentalking.core.types.frames import AudioChunk, VideoFrameData
from opentalking.media.frame_avatar import numpy_bgr_to_videoframe
from opentalking.models.registry import register_model

FPS = 25
log = logging.getLogger(__name__)


@dataclass
class FeatherTalkFeatures:
    vector: np.ndarray  # [frames, 40, 1024], including padded audio context
    frame_count: int


@dataclass
class FeatherTalkState:
    manifest: AvatarManifest
    frames: list[np.ndarray]
    inputs: list[Any]
    crops: list[np.ndarray]
    boxes: list[tuple[int, int, int, int]]
    model: Any
    hubert: Any
    extra: dict[str, Any] = field(default_factory=dict)


def _path(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _frame_index(index: int, count: int) -> int:
    if count == 1:
        return 0
    position = index % (2 * count - 2)
    return min(position, 2 * count - 2 - position)


@register_model("feathertalk")
class FeatherTalkAdapter:
    model_type = "feathertalk"

    def __init__(self) -> None:
        self._device = os.getenv("OPENTALKING_FEATHERTALK_DEVICE", "auto")
        self._hubert: Any = None
        self._model: Any = None

    def load_model(self, device: str = "cuda") -> None:
        # Generator weights can be person-specific and are loaded with the avatar.
        log.info("FeatherTalk initializing device: %s", device)
        import torch

        self._device = os.getenv("OPENTALKING_FEATHERTALK_DEVICE", device)
        if self._device == "auto":
            from .data_utils.feather_hubert.feather_hubert import get_best_device

            self._device = str(get_best_device())
        torch.empty(0, device=self._device)  # Fail early for unavailable devices.
        log.info("FeatherTalk device ready: %s", self._device)

    def load_avatar(self, avatar_path: str) -> FeatherTalkState:
        log.info("FeatherTalk loading avatar: %s", avatar_path)
        import cv2
        import torch

        from .data_utils.feather_hubert.feather_hubert import load_feather_hubert
        from .face_utils import (
            compute_face_bbox, crop_face, extract_inner, hwc_to_chw_tensor,
            mask_mouth, read_landmarks,
        )
        from .model import Model, load_checkpoint_state

        bundle = load_avatar_bundle(Path(avatar_path), strict=False)
        config = (bundle.manifest.metadata or {}).get("feathertalk", {})
        if not isinstance(config, dict):
            raise ValueError("avatar metadata.feathertalk must be an object")
        dataset = _path(str(config.get("dataset", "feathertalk")), bundle.path)
        checkpoint_raw = os.getenv("OPENTALKING_FEATHERTALK_CHECKPOINT") or config.get("checkpoint")
        if not checkpoint_raw:
            raise ValueError("Set OPENTALKING_FEATHERTALK_CHECKPOINT or metadata.feathertalk.checkpoint")
        checkpoint = _path(str(checkpoint_raw), bundle.path)
        hubert_raw = os.getenv("OPENTALKING_FEATHERTALK_HUBERT_CHECKPOINT") or config.get("hubert_checkpoint")
        hubert_checkpoint = (
            _path(str(hubert_raw), bundle.path) if hubert_raw
            else Path(__file__).with_name("feather_hubert.pth")
        )
        for path in (checkpoint, hubert_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"FeatherTalk checkpoint not found: {path}")
        paths = sorted((dataset / "full_body_img").glob("*.jpg"), key=lambda p: int(p.stem))
        if not paths:
            raise ValueError(f"No FeatherTalk frames in {dataset / 'full_body_img'}")
        self.load_model(self._device)
        log.info("FeatherTalk preparing %d avatar frames from %s", len(paths), dataset)
        frames, inputs, crops, boxes = [], [], [], []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read FeatherTalk frame: {path}")
            if image.shape[:2] != (bundle.manifest.height, bundle.manifest.width):
                raise ValueError(f"Frame dimensions do not match avatar manifest: {path}")
            box = compute_face_bbox(read_landmarks(str(dataset / "landmarks" / f"{path.stem}.lms")))
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                raise ValueError(f"Invalid FeatherTalk face bounding box {box}: {path}")
            crop = crop_face(image, box)
            inner = extract_inner(crop)
            inputs.append(torch.cat([
                hwc_to_chw_tensor(inner), hwc_to_chw_tensor(mask_mouth(inner.copy())),
            ], dim=0))
            frames.append(image)
            crops.append(crop)
            boxes.append(box)
            if len(frames) % 100 == 0:
                log.info("FeatherTalk prepared %d/%d avatar frames", len(frames), len(paths))
        log.info("FeatherTalk loading generator checkpoint: %s", checkpoint)
        model = Model().to(self._device)
        model.load_state_dict(load_checkpoint_state(str(checkpoint), self._device))
        model.eval()
        log.info("FeatherTalk loading HuBERT checkpoint: %s", hubert_checkpoint)
        hubert = load_feather_hubert(hubert_checkpoint, device=torch.device(self._device))
        self._model, self._hubert = model, hubert
        bundle.manifest.fps = FPS
        log.info("FeatherTalk avatar ready: %s (%d frames)", avatar_path, len(frames))
        return FeatherTalkState(bundle.manifest, frames, inputs, crops, boxes, model, hubert)

    def warmup(self, avatar_state: FeatherTalkState | None = None) -> None:
        import torch

        model = avatar_state.model if avatar_state is not None else self._model
        if model is None:
            return
        with torch.inference_mode():
            model(torch.zeros(1, 6, 144, 144, device=self._device),
                        torch.zeros(1, 40, 1024, device=self._device))

    @staticmethod
    def _pcm(chunk: AudioChunk) -> np.ndarray:
        data = np.asarray(chunk.data)
        if data.ndim not in (1, 2) or chunk.sample_rate <= 0:
            raise ValueError("FeatherTalk requires mono/stereo PCM with a positive sample rate")
        pcm = data.astype(np.float32)
        if np.issubdtype(data.dtype, np.signedinteger):
            pcm /= float(-np.iinfo(data.dtype).min)
        elif np.issubdtype(data.dtype, np.unsignedinteger):
            midpoint = float(np.iinfo(data.dtype).max + 1) / 2
            pcm = (pcm - midpoint) / midpoint
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=1)
        if chunk.sample_rate != 16000 and pcm.size:
            from scipy.signal import resample_poly
            from math import gcd

            divisor = gcd(chunk.sample_rate, 16000)
            pcm = resample_poly(pcm, 16000 // divisor, chunk.sample_rate // divisor)
        return pcm.astype(np.float32)

    def _features(self, pcm: np.ndarray, hubert: Any, start: int, count: int) -> FeatherTalkFeatures:
        import torch

        from .data_utils.feather_hubert.feather_hubert import get_feather_hubert_from_16k_speech
        from .face_utils import gather_audio_window, reshape_audio_feat

        if count == 0:
            return FeatherTalkFeatures(np.empty((0, 40, 1024), np.float32), 0)
        # Valid convolutions need 80 additional samples for exactly two tokens/frame.
        needed = (start + count) * 640 + 80
        pcm = np.pad(pcm, (0, max(0, needed - len(pcm))))
        hidden = get_feather_hubert_from_16k_speech(pcm, hubert, device=torch.device(self._device))
        features = hidden[: (len(pcm) - 80) // 640 * 2].numpy().reshape(-1, 2, 1024)
        vector = np.stack([
            reshape_audio_feat(gather_audio_window(features, i)).numpy()
            for i in range(start, start + count)
        ])
        return FeatherTalkFeatures(vector, count)

    def extract_features(self, audio_chunk: AudioChunk) -> FeatherTalkFeatures:
        if self._hubert is None:
            raise RuntimeError("Load a FeatherTalk avatar before extracting features")
        pcm = self._pcm(audio_chunk)
        return self._features(pcm, self._hubert, 0, (len(pcm) + 639) // 640)

    def extract_features_for_stream(self, audio_chunk: AudioChunk, avatar_state: FeatherTalkState) -> FeatherTalkFeatures:
        pcm = self._pcm(audio_chunk)
        extra = avatar_state.extra
        previous = np.asarray(extra.get("audio_context_pcm", np.empty(0)), dtype=np.float32)
        total_before = int(extra.get("audio_total_samples", 0))
        total = total_before + len(pcm)
        # Round cumulative time so repeated short chunks do not accumulate drift.
        start = (total_before + 639) // 640
        count = (total + 639) // 640 - start
        joined = np.concatenate([previous, pcm])
        context_start = total_before - len(previous)
        features = self._features(joined, avatar_state.hubert, start - context_start // 640, count)
        # Keep aligned left context for both the encoder and 20-frame audio window.
        keep_start = max(0, (len(joined) // 640 - 25) * 640)
        extra["audio_context_pcm"] = joined[keep_start:].copy()
        extra["audio_total_samples"] = total
        return features

    def infer(self, features: FeatherTalkFeatures, avatar_state: FeatherTalkState) -> list[np.ndarray]:
        import torch

        start = int(avatar_state.extra.get("frame_index_start", 0))
        predictions = []
        with torch.inference_mode():
            for i, audio in enumerate(features.vector):
                index = _frame_index(start + i, len(avatar_state.frames))
                prediction = avatar_state.model(
                    avatar_state.inputs[index].unsqueeze(0).to(self._device),
                    torch.from_numpy(audio).unsqueeze(0).to(self._device),
                )[0]
                predictions.append((prediction.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
        return predictions

    def compose_frame(self, avatar_state: FeatherTalkState, frame_idx: int, prediction: Any) -> VideoFrameData:
        import cv2

        from .face_utils import FACE_BORDER, FACE_INNER_SIZE

        index = _frame_index(frame_idx, len(avatar_state.frames))
        image = avatar_state.frames[index].copy()
        if prediction is not None:
            crop = avatar_state.crops[index].copy()
            crop[FACE_BORDER:FACE_BORDER + FACE_INNER_SIZE,
                 FACE_BORDER:FACE_BORDER + FACE_INNER_SIZE] = prediction
            x1, y1, x2, y2 = avatar_state.boxes[index]
            image[y1:y2, x1:x2] = cv2.resize(crop, (x2 - x1, y2 - y1))
        return numpy_bgr_to_videoframe(image, frame_idx * 1000.0 / FPS)

    def idle_frame(self, avatar_state: FeatherTalkState, frame_idx: int) -> VideoFrameData:
        return numpy_bgr_to_videoframe(avatar_state.frames[0].copy(), frame_idx * 1000.0 / FPS)
