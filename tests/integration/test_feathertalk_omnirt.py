"""Optional real-weight integration; configure OMNIRT_FEATHERTALK_* paths.

Run with PYTHONPATH=/path/to/omnirt/src and the three model asset variables.
"""
import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('omnirt')
from fastapi.testclient import TestClient
from omnirt.server.avatar_app import create_avatar_app

from opentalking.providers.synthesis.feathertalk.ws_client import FeatherTalkWSClient
from opentalking.providers.synthesis.audio2video_client import OmniRTAudio2VideoClient


@pytest.mark.asyncio
async def test_real_feathertalk_client_round_trip(monkeypatch):
    for name in ('DATASET', 'CHECKPOINT', 'HUBERT_CHECKPOINT'):
        if not os.getenv(f'OMNIRT_FEATHERTALK_{name}'):
            pytest.skip('Configure FeatherTalk asset paths to enable real-weight integration')
    import torch
    monkeypatch.setenv('OMNIRT_FEATHERTALK_RUNTIME', '1')
    monkeypatch.setenv('OMNIRT_FEATHERTALK_DEVICE', 'cpu')
    app = create_avatar_app()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    class Bridge:
        def __init__(self, socket):
            self.socket = socket
        async def send(self, message):
            if isinstance(message, str):
                self.socket.send_text(message)
            else:
                self.socket.send_bytes(message)
        async def recv(self):
            message = await asyncio.to_thread(self.socket.receive)
            return message.get('text') if message.get('text') is not None else message['bytes']
        async def close(self):
            pass

    try:
        with TestClient(app) as service, service.websocket_connect('/v1/audio2video/feathertalk') as socket:
            ws = FeatherTalkWSClient('ws://test/v1/audio2video/feathertalk')
            ws._ws = Bridge(socket)
            client = OmniRTAudio2VideoClient(ws)
            ref = Path(os.environ['OMNIRT_FEATHERTALK_DATASET']) / 'full_body_img/0.jpg'
            response = await client.init_session(ref_image=ref)
            assert response['chunk_samples'] == 640
            client.reset_audio_stream()
            rng = np.random.default_rng(8)
            for _ in range(3):
                client.set_audio_lookahead(rng.integers(-5000, 5000, 6400, dtype=np.int16))
                frames = await client.generate(rng.integers(-5000, 5000, 640, dtype=np.int16))
                assert len(frames) == 1
                assert frames[0].data.shape == (response['height'], response['width'], 3)
            idle, frames = await asyncio.gather(client.get_idle_frame(1), client.generate(np.zeros(640, np.int16)))
            assert idle.width == response['width']
            assert len(frames) == 1
            await client.close()
            assert app.state.realtime_avatar_service.runtime.feathertalk._states == {}
    finally:
        torch.set_num_threads(old_threads)
