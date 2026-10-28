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


def _write_frame(image_dir, landmark_dir, stem, value=0):
    import cv2

    image_dir.mkdir(parents=True, exist_ok=True)
    landmark_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / f'{stem}.jpg'
    cv2.imwrite(str(path), np.full((180, 180, 3), value, np.uint8))
    points = np.zeros((68, 2))
    points[1] = [10, 0]
    points[31] = [170, 0]
    points[52] = [0, 10]
    np.savetxt(landmark_dir / f'{stem}.lms', points)
    return path


def test_infer_compose_pipeline(tmp_path, monkeypatch):
    import cv2
    from opentalking.models.feathertalk import face_utils
    from opentalking.pipeline.speak.render_pipeline import _infer_predictions

    adapter = FeatherTalkAdapter()
    adapter._device = 'cpu'
    paths = [_write_frame(tmp_path / 'images', tmp_path / 'landmarks', str(i * 10), i) for i in range(3)]
    state = FeatherTalkState(
        manifest=SimpleNamespace(fps=25, width=180, height=180),
        frame_paths=paths, landmark_dir=tmp_path / 'landmarks',
        model=lambda image, audio: torch.ones(1, 3, 144, 144), hubert=None,
        extra={'frame_index_start': 2},
    )
    images_read, landmarks_read = [], []
    imread, read_landmarks = cv2.imread, face_utils.read_landmarks

    def read_image(path):
        images_read.append(path)
        return imread(path)

    def read_points(path):
        landmarks_read.append(path)
        return read_landmarks(path)

    monkeypatch.setattr(cv2, 'imread', read_image)
    monkeypatch.setattr(face_utils, 'read_landmarks', read_points)
    features = FeatherTalkFeatures(np.zeros((4, 40, 1024), np.float32), 4)
    predictions = _infer_predictions(adapter, state, features, frame_index_start=2, infer_batch_frames=2)
    assert images_read == [str(paths[i]) for i in [2, 1, 0, 1]]
    assert landmarks_read == [str(state.landmark_dir / f'{paths[i].stem}.lms') for i in [2, 1, 0, 1]]
    frame = adapter.compose_frame(state, 2, predictions[0])
    assert len(images_read) == 4  # Composition reuses the inferred image.
    assert frame.timestamp_ms == 80
    assert frame.data[20, 20, 0] == 255
    assert imread(str(paths[2]))[20, 20, 0] == 2
    assert adapter.idle_frame(state, 9).data[20, 20, 0] == 1
    assert adapter.compose_frame(state, 2, None).data[20, 20, 0] == 2
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
    def unexpected_read(*args, **kwargs):
        raise AssertionError('load_avatar must not read images or landmarks')

    from opentalking.models.feathertalk import face_utils
    with monkeypatch.context() as loading:
        loading.setattr(cv2, 'imread', unexpected_read)
        loading.setattr(face_utils, 'read_landmarks', unexpected_read)
        state = adapter.load_avatar(str(tmp_path))
    assert state.frame_paths == [dataset / 'full_body_img' / '0.jpg']
    assert state.landmark_dir == dataset / 'landmarks'
    assert state.manifest.fps == 25
    assert loaded == [{'path': str(tmp_path / 'generator.pth')}]
    points[31, 0] = 200
    np.savetxt(dataset / 'landmarks' / '0.lms', points)
    features = FeatherTalkFeatures(np.zeros((1, 40, 1024), np.float32), 1)
    with pytest.raises(ValueError, match='bounding box'):
        adapter.infer(features, state)


@pytest.mark.asyncio
async def test_feathertalk_prewarm_accepts_avatar_state():
    from opentalking.providers.synthesis.audio2video_client import LocalAudio2VideoClient

    adapter = FeatherTalkAdapter()
    adapter._device = 'cpu'
    calls = []

    def model(image, audio):
        calls.append((tuple(image.shape), tuple(audio.shape)))

    client = LocalAudio2VideoClient(adapter, device='cpu')
    client.avatar_state = SimpleNamespace(model=model)
    try:
        assert await client.prewarm() == {'type': 'prewarm_ok'}
    finally:
        await client.close()
    assert calls == [((1, 6, 144, 144), (1, 40, 1024))]
