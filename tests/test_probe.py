"""Probe construction and search-input routing.

Uses stub backends so these run without downloading any model.
"""

import cv2
import numpy as np
import pytest

from hhg3 import pipeline
from hhg3.config import Config
from hhg3.face.detect import Detection


class StubDetector:
    name = "stub-detector"

    def __init__(self, detections):
        self._detections = detections

    def detect(self, image):
        return list(self._detections)


class StubEmbedder:
    name = "stub-embedder"
    identity_grade = True
    default_threshold = 0.4

    def embed(self, image, det):
        return np.ones(8, dtype=np.float32) / np.sqrt(8)


@pytest.fixture
def photo(tmp_path):
    rng = np.random.default_rng(3)
    path = tmp_path / "probe.jpg"
    cv2.imwrite(str(path), rng.integers(0, 255, (400, 600, 3), dtype=np.uint8))
    return path


FACE = Detection(bbox=(250, 150, 100, 120), score=0.9, landmarks=np.zeros((5, 2), np.float32))


def test_face_probe_carries_bbox_and_embedding(photo, tmp_path):
    probe = pipeline.scan_probe(photo, StubDetector([FACE]), StubEmbedder(), Config(), tmp_path)
    assert probe.kind == "face" and probe.has_face
    assert probe.bbox == FACE.bbox
    assert len(probe.embedding) == 8
    assert probe.crop_path and probe.crop_sha256


def test_probe_falls_back_to_whole_image_when_no_face(photo, tmp_path):
    probe = pipeline.scan_probe(photo, StubDetector([]), StubEmbedder(), Config(), tmp_path)
    assert probe.kind == "image" and not probe.has_face
    assert probe.bbox is None and probe.embedding == [] and probe.crop_path is None


def test_require_face_aborts_instead_of_falling_back(photo, tmp_path):
    cfg = Config()
    cfg.require_face = True
    with pytest.raises(pipeline.PipelineError, match="no face detected"):
        pipeline.scan_probe(photo, StubDetector([]), StubEmbedder(), cfg, tmp_path)


def test_auto_sends_the_padded_face_box(photo, tmp_path):
    probe = pipeline.scan_probe(photo, StubDetector([FACE]), StubEmbedder(), Config(), tmp_path)
    query, described = pipeline.search_input(probe, Config(), tmp_path)
    assert query.name == "search_crop.jpg"
    assert "face box" in described


def test_auto_sends_the_whole_image_when_there_is_no_face(photo, tmp_path):
    probe = pipeline.scan_probe(photo, StubDetector([]), StubEmbedder(), Config(), tmp_path)
    query, described = pipeline.search_input(probe, Config(), tmp_path)
    assert query == photo and described == "whole image"


def test_source_mode_ignores_the_face(photo, tmp_path):
    cfg = Config()
    cfg.search_image = "source"
    probe = pipeline.scan_probe(photo, StubDetector([FACE]), StubEmbedder(), cfg, tmp_path)
    query, _ = pipeline.search_input(probe, cfg, tmp_path)
    assert query == photo


def test_face_mode_without_a_face_is_an_error(photo, tmp_path):
    cfg = Config()
    cfg.search_image = "face"
    probe = pipeline.scan_probe(photo, StubDetector([]), StubEmbedder(), Config(), tmp_path)
    with pytest.raises(pipeline.PipelineError, match="no face was detected"):
        pipeline.search_input(probe, cfg, tmp_path)


@pytest.mark.parametrize("bbox", [(0, 0, 80, 90), (560, 340, 40, 60), (250, 150, 100, 120)])
def test_padding_never_leaves_the_image(bbox):
    """Padding must be more picture, never black bars, so the crop is clamped."""
    image = np.zeros((400, 600, 3), np.uint8)
    crop = pipeline._crop(image, bbox, pad_frac=2.0)
    assert crop.size > 0
    assert crop.shape[0] <= image.shape[0] and crop.shape[1] <= image.shape[1]


def test_bigger_padding_gives_a_bigger_crop():
    image = np.zeros((400, 600, 3), np.uint8)
    tight = pipeline._crop(image, (250, 150, 100, 120), 0.25)
    wide = pipeline._crop(image, (250, 150, 100, 120), 1.0)
    assert wide.shape[0] > tight.shape[0] and wide.shape[1] > tight.shape[1]
