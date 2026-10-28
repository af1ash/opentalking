import asyncio
import base64
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from opentalking.core.config import Settings
from opentalking.core.model_config import clear_model_config_cache
from opentalking.core.types.frames import VideoFrameData
from opentalking.pipeline.speak.synthesis_runner import FlashTalkRunner
from opentalking.providers.synthesis.audio2video_client import OmniRTAudio2VideoClient
from opentalking.providers.synthesis.feathertalk.ws_client import FeatherTalkWSClient


def video_payload():
    from PIL import Image
    import io
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8), (40, 50, 60)).save(buffer, format='JPEG')
    jpeg = buffer.getvalue()
    return b'VIDX' + struct.pack('<II', 1, len(jpeg)) + jpeg


class Socket:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
    async def send(self, payload):
        self.sent.append(payload)
        await asyncio.sleep(0)
    async def recv(self):
        await asyncio.sleep(0)
        reply = self.replies.pop(0)
        return json.dumps(reply) if isinstance(reply, dict) else reply
    async def close(self):
        pass


@pytest.mark.asyncio
async def test_init_forwards_remote_paths():
    client = FeatherTalkWSClient('ws://omnirt/v1/audio2video/feathertalk')
    client._ws = Socket([{'type': 'init_ok', 'frame_num': 3, 'motion_frames_num': 1,
                          'slice_len': 1, 'fps': 25, 'width': 8, 'height': 8, 'chunk_samples': 640}])
    result = await client.init_session(b'image', video_config={'feathertalk_dataset': '/remote/data', 'feathertalk_checkpoint': '/remote/model.pth'})
    sent = json.loads(client._ws.sent[0])
    assert sent['feathertalk_dataset'] == '/remote/data'
    assert sent['feathertalk_checkpoint'] == '/remote/model.pth'
    assert result['chunk_samples'] == 640


@pytest.mark.asyncio
async def test_context_generate_idle_requests_are_serialized():
    client = FeatherTalkWSClient('ws://omnirt/v1/audio2video/feathertalk')
    socket = client._ws = Socket([{'type': 'audio_context_ok'}, video_payload(), video_payload()])
    future = np.arange(6400, dtype=np.int16)
    client.set_audio_lookahead(future)
    frames, idle = await asyncio.gather(client.generate(np.zeros(640, np.int16)), client.get_idle_frame(2))
    assert len(frames) == 1 and idle.width == 8
    context = json.loads(socket.sent[0])
    assert context['reset'] is True
    assert base64.b64decode(context['pcm_s16le']) == future.astype('<i2').tobytes()
    assert socket.sent[1] == b'AUDI' + b'\0\0' * 640
    assert json.loads(socket.sent[2]) == {'type': 'idle', 'frame_index': 2}
    socket.replies.extend([{'type': 'audio_context_ok'}, video_payload()])
    client.reset_audio_stream()
    await client.generate(np.zeros(640, np.int16))
    assert json.loads(socket.sent[3])['reset'] is True


@pytest.mark.asyncio
async def test_protocol_errors_are_reported():
    client = FeatherTalkWSClient('ws://omnirt/v1/audio2video/feathertalk')
    client._ws = Socket([{'type': 'error', 'message': 'checkpoint not found'}])
    with pytest.raises(RuntimeError, match='checkpoint not found'):
        await client.generate(np.zeros(640, np.int16))
    with pytest.raises(ValueError, match='640 PCM'):
        await client.generate(np.zeros(641, np.int16))
    with pytest.raises(ValueError, match='400ms'):
        client.set_audio_lookahead(np.zeros(6401, np.int16))
    client._ws = Socket([b'VIDX' + struct.pack('<II', 1, 200)])
    with pytest.raises(RuntimeError, match='exactly one JPEG'):
        await client.get_idle_frame(0)


@pytest.mark.asyncio
async def test_runner_lookahead_handles_40ms_chunks_and_tail():
    client = SimpleNamespace()
    resets, contexts, output = [], [], []
    client.reset_audio_stream = lambda: resets.append(True)
    client.set_audio_lookahead = lambda pcm: contexts.append(pcm.copy())
    runner = object.__new__(FlashTalkRunner)
    runner.model_type = 'feathertalk'
    runner.audio2video = client
    queue = asyncio.Queue(maxsize=8)
    chunks = [np.full(640, i, np.int16) for i in range(15)]
    async def produce():
        for i, chunk in enumerate(chunks):
            await queue.put((chunk, str(i)))
        await queue.put(None)
    async def consume():
        async for pcm, tag in runner._iter_model_audio_chunks(queue):
            output.append((pcm, tag))
    await asyncio.wait_for(asyncio.gather(produce(), consume()), timeout=2)
    assert resets == [True]
    assert [tag for _, tag in output] == [str(i) for i in range(15)]
    np.testing.assert_array_equal(np.concatenate([pcm for pcm, _ in output]), np.concatenate(chunks))
    np.testing.assert_array_equal(contexts[0], np.concatenate(chunks[1:11]))
    assert [len(pcm) for pcm in contexts[-3:]] == [640, 0, 0]


def test_remote_model_paths_are_not_resolved_on_api_host(tmp_path, monkeypatch):
    import opentalking.pipeline.speak.synthesis_runner as module
    runner = object.__new__(FlashTalkRunner)
    runner.model_type = 'feathertalk'
    runner._avatar_manifest_metadata = lambda: {'feathertalk': {'dataset': '/remote/person'}}
    monkeypatch.setattr(module, 'get_settings', lambda: SimpleNamespace())
    clear_model_config_cache()
    config = runner._feathertalk_video_config()
    assert config['feathertalk_dataset'] == '/remote/person'
    assert config['chunk_samples'] == 640
    monkeypatch.setenv('OPENTALKING_FEATHERTALK_DATASET', 'server-relative/person')
    assert runner._feathertalk_video_config()['feathertalk_dataset'] == 'server-relative/person'
    clear_model_config_cache()


def test_settings_reads_feathertalk_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENTALKING_FEATHERTALK_BACKEND', raising=False)
    env = tmp_path / '.env'
    env.write_text('OPENTALKING_FEATHERTALK_BACKEND=direct_ws\nOPENTALKING_FEATHERTALK_WS_URL=ws://inference/v1/audio2video/feathertalk\n')
    settings = Settings(_env_file=env)
    assert settings.feathertalk_backend == 'direct_ws'
    assert settings.feathertalk_ws_url == 'ws://inference/v1/audio2video/feathertalk'


@pytest.mark.asyncio
async def test_remote_idle_is_used_in_runner():
    runner = object.__new__(FlashTalkRunner)
    runner.model_type = 'feathertalk'
    runner.webrtc = SimpleNamespace(draining=False)
    runner._quicktalk_idle_video = None
    runner._idle_frames = []
    runner._last_frame = None
    runner._idle_frame_idx = 0
    runner._ensure_media_clock_started = lambda: None
    seen, output = [], []
    async def idle(index):
        seen.append(index)
        return VideoFrameData(np.full((2, 2, 3), index, np.uint8), 2, 2, index * 40)
    async def put(frame):
        output.append(frame)
    runner.audio2video = SimpleNamespace(get_idle_frame=idle)
    runner._video_put_safe = put
    await runner._idle_tick()
    await runner._idle_tick()
    assert seen == [0, 1]
    assert [int(frame.data[0, 0, 0]) for frame in output] == [0, 1]


def test_worker_uses_feathertalk_remote_client(monkeypatch):
    from pathlib import Path
    from opentalking.runtime import task_consumer
    from opentalking.core.in_memory_redis import InMemoryRedis
    captured = {}
    monkeypatch.setattr(task_consumer, 'FlashTalkRunner', lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(task_consumer, 'resolve_model_backend', lambda *_: SimpleNamespace(backend='omnirt', ws_url=''))
    monkeypatch.setattr('opentalking.providers.synthesis.omnirt.resolve_synthesis_ws_url', lambda *_: 'ws://inference/v1/audio2video/feathertalk')
    task_consumer._create_runner({'model': 'feathertalk', 'avatar_id': 'test', 'session_id': 'sess_test'}, InMemoryRedis(), Path('.'), 'cpu')
    assert captured['model_type'] == 'feathertalk'
    assert isinstance(captured['audio2video_client'], OmniRTAudio2VideoClient)
    assert isinstance(captured['audio2video_client']._ws_client, FeatherTalkWSClient)
