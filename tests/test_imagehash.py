import cv2
import numpy as np
import pytest

from hhg3.imagehash import agreement, phash


@pytest.fixture
def picture():
    """Something with structure - a flat image has no meaningful DCT signature."""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    return cv2.resize(base, (480, 480), interpolation=cv2.INTER_LINEAR)


def test_hash_is_64_bits(picture):
    assert phash(picture).size == 64


def test_identical_images_agree_completely(picture):
    assert agreement(phash(picture), phash(picture)) == 1.0


def test_survives_rescaling(picture):
    small = cv2.resize(picture, (160, 160), interpolation=cv2.INTER_AREA)
    assert agreement(phash(picture), phash(small)) >= 0.9


def test_survives_heavy_jpeg_recompression(picture):
    _, buf = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 35])
    assert agreement(phash(picture), phash(cv2.imdecode(buf, cv2.IMREAD_COLOR))) >= 0.9


def test_unrelated_images_land_near_chance(picture):
    rng = np.random.default_rng(99)
    other = cv2.resize(
        rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), (480, 480), interpolation=cv2.INTER_LINEAR
    )
    # Chance agreement on a 64-bit hash is 0.5; the default threshold is 0.85.
    assert agreement(phash(picture), phash(other)) < 0.85


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        agreement(np.zeros(64, bool), np.zeros(32, bool))
