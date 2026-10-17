"""Regression tests for the training pipeline. Run with:  python -m pytest tests

They build tiny synthetic simulator recordings, so no dataset download is needed.
"""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from selfdriver.preprocess import decode_rgb, load_rgb  # noqa: E402


def make_recording(folder, n_frames=60, seed=0):
    """Write a fake Udacity-simulator recording: IMG/*.jpg + driving_log.csv
    with Windows-style absolute paths, like the real simulator produces."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(folder, "IMG"))
    rows = []
    for i in range(n_frames):
        names = [f"{cam}_2026_01_01_00_00_00_{i:03d}.jpg" for cam in ("center", "left", "right")]
        for name in names:
            img = rng.integers(0, 255, (160, 320, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(folder, "IMG", name), img)
        steering = 0.0 if i % 3 else rng.uniform(-1, 1)
        rows.append([f"C:\\Users\\me\\rec\\IMG\\{n}" for n in names] + [steering, 0, 0, 10])
    pd.DataFrame(rows).to_csv(os.path.join(folder, "driving_log.csv"), header=False, index=False)


def run_train(*args):
    result = subprocess.run(
        [sys.executable, "train.py", *args], cwd=ROOT, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


# --- issue #3: validation must not contain near-copies of training frames ----------


def test_split_by_blocks_keeps_neighbouring_frames_together():
    df = pd.DataFrame({"frame": np.arange(1000), "steering": 0.0})
    val, tr = train.split_by_blocks(df, val_split=0.2, block_size=50, rng=np.random.default_rng(0))
    assert len(val) + len(tr) == 1000
    assert len(val) == 200
    # Every block of 50 consecutive frames is entirely on one side of the split.
    assert set(val["frame"] // 50).isdisjoint(set(tr["frame"] // 50))


# --- issue #4: balancing cap scales with the amount of data -------------------------


def test_balance_auto_cap_keeps_share_of_extra_data():
    rng = np.random.default_rng(0)
    steering = np.where(rng.random(4000) < 0.7, 0.0, rng.normal(0, 0.3, 4000).clip(-1, 1))
    small = pd.DataFrame({"steering": steering})
    big = pd.concat([small] * 6, ignore_index=True)
    kept_small = len(train.balance(small, 25, None, np.random.default_rng(0))) / len(small)
    kept_big = len(train.balance(big, 25, None, np.random.default_rng(0))) / len(big)
    assert abs(kept_small - kept_big) < 0.02


def test_balance_explicit_cap_is_respected():
    df = pd.DataFrame({"steering": np.zeros(1000)})
    assert len(train.balance(df, 25, 100, np.random.default_rng(0))) == 100


# --- issue #5: images are kept compressed and decoded on use ------------------------


def test_decode_rgb_matches_load_rgb(tmp_path):
    path = str(tmp_path / "frame.jpg")
    cv2.imwrite(path, np.random.default_rng(0).integers(0, 255, (160, 320, 3), dtype=np.uint8))
    jpeg = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    np.testing.assert_array_equal(decode_rgb(jpeg), load_rgb(path))


# --- end to end: load_log path handling, issues #2 and #6 ---------------------------


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    folder = str(tmp_path_factory.mktemp("rec"))
    make_recording(folder)
    return folder


def test_load_log_finds_images_from_windows_paths(recording):
    df = train.load_log(recording)
    assert len(df) == 60
    assert all(os.path.exists(p) for p in df["center"])


def test_learning_rate_logged_for_the_epoch_it_was_used(tmp_path):
    # Force a plateau (min_delta is huge, so nothing counts as an improvement):
    # ReduceLROnPlateau halves the rate at the end of epoch 1, so epochs 0 and 1
    # trained at 1e-3 and epoch 2 at 5e-4.
    from tensorflow import keras

    model = keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    x, y = np.zeros((8, 2)), np.zeros(8)
    logger = train.EpochLogger(str(tmp_path / "log.csv"), pd.DataFrame())
    reduce = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_delta=10)
    model.fit(x, y, validation_data=(x, y), epochs=3, callbacks=[reduce, logger], verbose=0)
    np.testing.assert_allclose(logger.log["learning_rate"], [1e-3, 1e-3, 5e-4], rtol=1e-5)
    assert len(pd.read_csv(tmp_path / "log.csv")) == 3


def test_resume_from_model_outside_out_without_improvement(recording, tmp_path):
    common = ["--data", recording, "--steps-per-epoch", "1", "--batch-size", "4", "--val-block", "5"]
    first = str(tmp_path / "first")
    run_train(*common, "--out", first, "--epochs", "1")
    second = str(tmp_path / "second")
    run_train(*common, "--out", second, "--epochs", "1", "--lr", "1e-12", "--resume", os.path.join(first, "model.keras"))
    assert os.path.exists(os.path.join(second, "model.keras"))
    with open(os.path.join(second, "training_summary.json")) as f:
        assert json.load(f)["epochs_total"] == 1
