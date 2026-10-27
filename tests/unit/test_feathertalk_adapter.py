from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opentalking.core.types.frames import AudioChunk
from opentalking.models.feathertalk.adapter import (
    FeatherTalkAdapter, FeatherTalkFeatures, FeatherTalkState, _frame_index,
)


def test_registration():
    from opentalking.models.registry import get_adapter
    from opentalking.core.interfaces.model_adapter import ModelAdapter

    assert isinstance(get_adapter('feathertalk'), ModelAdapter)


def test_pcm_resampling_and_stereo():
    pcm = FeatherTalkAdapter._pcm(AudioChunk(np.full((4800, 2), 16384, np.int16), 48000, 100))
    assert pcm.shape == (1600,)
    assert pcm[800] == pytest.approx(.5, abs=.001)


def test_audio_windows(monkeypatch):
    from opentalking.models.feathertalk.data_utils.feather_hubert import feather_hubert

    def extract(pcm, model, device):
        return torch.ones(((len(pcm) - 80) // 320, 1024))

    monkeypatch.setattr(feather_hubert, 'get_feather_hubert_from_16k_speech', extract)
    adapter = FeatherTalkAdapter()
    adapter._device = 'cpu'
    adapter._hubert = object()
    features = adapter.extract_features(AudioChunk(np.zeros(1600, np.int16), 16000, 100))
    assert features.vector.shape == (3, 40, 1024)
    assert np.all(features.vector[0, :20] == 0)
    assert np.all(features.vector[0, 20:26] == 1)
    assert adapter.extract_features(AudioChunk(np.empty(0), 16000, 0)).frame_count == 0


def test_stream_timing_and_reset(monkeypatch):
    adapter = FeatherTalkAdapter()
    seen = []

    def extract(pcm, hubert, start, count):
        seen.append((len(pcm), start, count))
        return FeatherTalkFeatures(np.empty((count, 40, 1024), np.float32), count)

    monkeypatch.setattr(adapter, '_features', extract)
    state = SimpleNamespace(extra={}, hubert=object())
    counts = []
    for _ in range(100):
        counts.append(adapter.extract_features_for_stream(AudioChunk(np.zeros(321, np.int16), 16000, 20), state).frame_count)
    assert sum(counts) == 51
    assert len(state.extra['audio_context_pcm']) < 26 * 640
    from opentalking.pipeline.speak.render_pipeline import reset_avatar_speech_state
    reset_avatar_speech_state(state)
    assert adapter.extract_features_for_stream(AudioChunk(np.zeros(321, np.int16), 16000, 20), state).frame_count == 1
    assert seen[-1] == (321, 0, 1)


def test_infer_compose_pipeline():
    from opentalking.pipeline.speak.render_pipeline import _infer_predictions

    adapter = FeatherTalkAdapter()
    adapter._device = 'cpu'
    frames = [np.full((180, 180, 3), i, np.uint8) for i in range(3)]
    state = FeatherTalkState(
        manifest=SimpleNamespace(fps=25), frames=frames,
        inputs=[torch.zeros(6, 144, 144)] * 3,
        crops=[np.zeros((152, 152, 3), np.uint8)] * 3,
        boxes=[(0, 0, 152, 152)] * 3,
        model=lambda image, audio: torch.ones(1, 3, 144, 144), hubert=None,
        extra={'frame_index_start': 2},
    )
    features = FeatherTalkFeatures(np.zeros((4, 40, 1024), np.float32), 4)
    predictions = _infer_predictions(adapter, state, features, frame_index_start=2, infer_batch_frames=2)
    frame = adapter.compose_frame(state, 2, predictions[0])
    assert frame.timestamp_ms == 80
    assert frame.data[20, 20, 0] == 255
    assert frames[2][20, 20, 0] == 2
    assert adapter.idle_frame(state, 9).data[20, 20, 0] == 0
    assert [_frame_index(i, 3) for i in range(6)] == [0, 1, 2, 1, 0, 1]


def test_load_avatar(tmp_path, monkeypatch):
    import cv2
    import json
    from opentalking.models.feathertalk import model
    from opentalking.models.feathertalk.data_utils.feather_hubert import feather_hubert

    dataset = tmp_path / 'feathertalk'
    (dataset / 'full_body_img').mkdir(parents=True)
    (dataset / 'landmarks').mkdir()
    cv2.imwrite(str(dataset / 'full_body_img' / '0.jpg'), np.zeros((180, 180, 3), np.uint8))
    points = np.zeros((68, 2))
    points[1] = [10, 0]
    points[31] = [170, 0]
    points[52] = [0, 10]
    np.savetxt(dataset / 'landmarks' / '0.lms', points)
    (tmp_path / 'generator.pth').touch()
    (tmp_path / 'hubert.pth').touch()
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'id': 'test', 'model_type': 'feathertalk', 'fps': 30,
        'sample_rate': 16000, 'width': 180, 'height': 180, 'version': '1.0',
        'metadata': {'feathertalk': {'checkpoint': 'generator.pth', 'hubert_checkpoint': 'hubert.pth'}},
    }))
    loaded = []

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state):
            loaded.append(state)

        def eval(self):
            return self

    monkeypatch.setenv('OPENTALKING_FEATHERTALK_DEVICE', 'cpu')
    monkeypatch.setattr(model, 'Model', FakeModel)
    monkeypatch.setattr(model, 'load_checkpoint_state', lambda path, device: {'path': path})
    monkeypatch.setattr(feather_hubert, 'load_feather_hubert', lambda path, device: object())
    adapter = FeatherTalkAdapter()
    state = adapter.load_avatar(str(tmp_path))
    assert state.inputs[0].shape == (6, 144, 144)
    assert state.manifest.fps == 25
    assert loaded == [{'path': str(tmp_path / 'generator.pth')}]
    points[31, 0] = 200
    np.savetxt(dataset / 'landmarks' / '0.lms', points)
    with pytest.raises(ValueError, match='bounding box'):
        adapter.load_avatar(str(tmp_path))
