from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import Any

import numpy as np

from opentalking.core.types.frames import VideoFrameData
from opentalking.providers.synthesis.flashtalk.ws_client import (
    MAGIC_VIDEO, FlashTalkWSClient, _decode_jpeg,
)


class FeatherTalkWSClient(FlashTalkWSClient):
    """OmniRT FeatherTalk: AUDI/VIDX plus audio_context and idle requests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._backend_name = "feathertalk"
        self._request_lock = asyncio.Lock()
        self._lookahead: bytes | None = None
        self._reset_pending = True

    async def init_session(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        async with self._request_lock:
            response = await super().init_session(*args, **kwargs)
            # if self.audio_chunk_samples != 640 or self.fps != 25:
            #     raise RuntimeError("FeatherTalk requires 640-sample audio chunks at 25fps")
            self._lookahead = None
            self._reset_pending = True
            return response

    def reset_audio_stream(self) -> None:
        self._reset_pending = True
        self._lookahead = None

    def set_audio_lookahead(self, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype='<i2').reshape(-1)
        # if pcm.size > 6400:
        #     raise ValueError("FeatherTalk audio lookahead must not exceed 400ms")
        self._lookahead = pcm.tobytes()

    async def generate(self, audio_pcm: np.ndarray) -> list[VideoFrameData]:
        pcm = np.asarray(audio_pcm, dtype=np.int16).reshape(-1)
        if pcm.size == 0:
            return []
        # if pcm.size != 640:
        #     raise ValueError("FeatherTalk requires exactly 640 PCM samples per request")
        async with self._request_lock:
            if self._ws is None:
                raise RuntimeError("Not connected. Call connect() first.")
            if self._lookahead is not None or self._reset_pending:
                await self._ws.send(json.dumps({
                    "type": "audio_context",
                    "pcm_s16le": base64.b64encode(self._lookahead or b'').decode('ascii'),
                    "reset": self._reset_pending,
                }))
                response = json.loads(await self._ws.recv())
                if response.get("type") != "audio_context_ok":
                    raise RuntimeError(f"FeatherTalk audio context failed: {response.get('message')}")
                self._reset_pending = False
                self._lookahead = None
            return await super().generate(pcm)

    async def get_idle_frame(self, frame_index: int) -> VideoFrameData:
        async with self._request_lock:
            if self._ws is None:
                raise RuntimeError("Not connected. Call connect() first.")
            await self._ws.send(json.dumps({"type": "idle", "frame_index": frame_index}))
            response = await self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"FeatherTalk idle failed: {json.loads(response).get('message')}")
            if len(response) < 12 or response[:4] != MAGIC_VIDEO:
                raise RuntimeError("Invalid FeatherTalk idle video payload")
            count, length = struct.unpack('<II', response[4:12])
            if count != 1 or length == 0 or length != len(response) - 12:
                raise RuntimeError("FeatherTalk idle response must contain exactly one JPEG frame")
            image = _decode_jpeg(response[12:])
            return VideoFrameData(image, image.shape[1], image.shape[0], frame_index * 1000.0 / self.fps)

    async def close(self, send_close_msg: bool = True) -> None:
        async with self._request_lock:
            await super().close(send_close_msg=send_close_msg)
            self._lookahead = None
            self._reset_pending = True
