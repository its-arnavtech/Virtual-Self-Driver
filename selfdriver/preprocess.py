"""Turning camera frames into the numbers the network sees.

`preprocess` must be applied identically at training time and at driving time,
so both train.py and drive.py import it from here.
"""

import cv2
import numpy as np

# Udacity simulator frames are 160x320 RGB.
CROP_TOP = 60  # sky, trees, hills: nothing that tells you how to steer
CROP_BOTTOM = 135  # the car's own bonnet
OUT_W, OUT_H = 200, 66  # NVIDIA input size


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """160x320 RGB uint8 -> 66x200 YUV uint8 (normalization happens inside the model)."""
    img = rgb[CROP_TOP:CROP_BOTTOM, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Augmentation: random, label-preserving changes so the network sees far more
# variety than the raw recording contains (and can't just memorise it).
# ---------------------------------------------------------------------------


def _zoom(img, rng):
    scale = rng.uniform(1.0, 1.3)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _pan(img, rng):
    h, w = img.shape[:2]
    tx, ty = rng.uniform(-0.1, 0.1) * w, rng.uniform(-0.1, 0.1) * h
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _brightness(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.3, 1.2), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment(rgb: np.ndarray, angle: float, rng: np.random.Generator):
    """Each transform is applied with 50% probability."""
    img = rgb
    if rng.random() < 0.5:
        img = _pan(img, rng)
    if rng.random() < 0.5:
        img = _zoom(img, rng)
    if rng.random() < 0.5:
        img = _brightness(img, rng)
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)  # mirror the road...
        angle = -angle  # ...so the correct steering mirrors too
    return img, angle
