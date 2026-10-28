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


@pytest.mark.asyncio
async def test_live_audio_lookahead_preserves_audio_and_subtitles():
    import asyncio

    runner = object.__new__(FlashTalkRunner)
    runner.model_type = 'feathertalk'
    client = LocalAudio2VideoClient(FeatherTalkAdapter(), device='cpu')
    client.avatar_state = SimpleNamespace(extra={'audio_total_samples': 99})
    client.speech_frame_index = 5
    runner.audio2video = client
    queue = asyncio.Queue()
    # Short chunks require collecting more than one following queue item.
    chunks = [np.full(3200, i, np.int16) for i in range(4)]
    for i, pcm in enumerate(chunks):
        await queue.put((pcm, str(i)))
    await queue.put(None)
    output, futures = [], []
    try:
        async for pcm, subtitle in runner._iter_model_audio_chunks(queue):
            output.append((pcm, subtitle))
            futures.append(client.avatar_state.extra['feathertalk_lookahead_pcm'].copy())
        assert client.speech_frame_index == 0
        assert client.avatar_state.extra['audio_total_samples'] == 0
        assert [subtitle for _, subtitle in output] == ['0', '1', '2', '3']
        np.testing.assert_array_equal(np.concatenate([pcm for pcm, _ in output]), np.concatenate(chunks))
        np.testing.assert_array_equal(futures[0], np.concatenate(chunks[1:3]).astype(np.float32) / 32768)
        assert [len(pcm) for pcm in futures] == [6400, 6400, 3200, 0]
    finally:
        await client.close()
