from types import SimpleNamespace
import threading

import numpy as np
import pytest

from opentalking.models.feathertalk.adapter import FeatherTalkAdapter
from opentalking.pipeline.speak.synthesis_runner import FlashTalkRunner
from opentalking.providers.synthesis.audio2video_client import LocalAudio2VideoClient


@pytest.mark.asyncio
@pytest.mark.parametrize('frame_count', [1, 3])
async def test_runner_plays_feathertalk_idle_through_local_executor(monkeypatch, frame_count):
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(frame_count)]
    calls = []
    main_thread = threading.get_ident()

    def read_frame(state, index):
        calls.append(threading.get_ident())
        return frames[index].copy()

    monkeypatch.setattr(FeatherTalkAdapter, '_read_frame', staticmethod(read_frame))
    client = LocalAudio2VideoClient(FeatherTalkAdapter(), device='cpu')
    client.avatar_state = SimpleNamespace(frame_paths=list(range(frame_count)))
    client.frame_index = 42
    client.speech_frame_index = 7
    runner = object.__new__(FlashTalkRunner)
    runner.model_type = 'feathertalk'
    runner.audio2video = client
    runner.webrtc = SimpleNamespace(draining=False)
    runner._idle_frame_idx = 0
    runner._quicktalk_idle_video = None
    runner._idle_frames = []
    runner._last_frame = np.full((2, 2, 3), 99, np.uint8)
    runner._ensure_media_clock_started = lambda: None
    output = []

    async def put(frame):
        output.append(int(frame.data[0, 0, 0]))

    runner._video_put_safe = put
    try:
        for _ in range(6):
            await runner._idle_tick()
        assert output == ([0, 1, 2, 1, 0, 1] if frame_count == 3 else [0] * 6)
        assert calls and all(thread != main_thread for thread in calls)
        assert runner._idle_frame_idx == 6
        assert client.frame_index == 42
        assert client.speech_frame_index == 7
        runner.webrtc.draining = True
        await runner._idle_tick()
        assert len(output) == 6
        assert len(calls) == 6
    finally:
        await client.close()
