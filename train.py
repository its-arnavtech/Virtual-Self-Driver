"""Step 3: TRAIN. Teach the NVIDIA network to steer from recorded driving.

    python train.py                         # uses data/track (scripts/download_data.sh)
    python train.py --data path/to/my_sim_recording --epochs 30

A data folder is whatever the Udacity simulator writes in Training Mode:
    driving_log.csv   center,left,right,steering,throttle,brake,speed
    IMG/              center_*.jpg, left_*.jpg, right_*.jpg
Several folders can be passed to --data and they are combined.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from selfdriver.model import build_nvidia_model
from selfdriver.preprocess import augment, load_rgb, preprocess

COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


def load_log(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "driving_log.csv")
    df = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True)
    if df.iloc[0]["center"] == "center":  # Udacity's sample data ships with a header row
        df = df.iloc[1:]
    df["steering"] = df["steering"].astype(float)
    # The simulator stores absolute paths from the recording machine (often
    # Windows paths). Keep only the file name and look for it in this IMG folder.
    for cam in ("center", "left", "right"):
        df[cam] = df[cam].map(lambda p: os.path.join(data_dir, "IMG", p.replace("\\", "/").split("/")[-1]))
    return df.reset_index(drop=True)


def balance(df: pd.DataFrame, bins: int, max_per_bin: int, rng) -> pd.DataFrame:
    """Humans drive straight most of the time, so most rows have steering == 0.
    Left as is, the network learns that "always go straight" is nearly always
    right. Cap how many samples any steering range may contribute."""
    edges = np.linspace(-1, 1, bins + 1)
    which = np.clip(np.digitize(df["steering"], edges) - 1, 0, bins - 1)
    keep = []
    for b in range(bins):
        idx = np.flatnonzero(which == b)
        rng.shuffle(idx)
        keep.extend(idx[:max_per_bin])
    return df.iloc[sorted(keep)].reset_index(drop=True)


def expand_cameras(df: pd.DataFrame, correction: float):
    """The left/right cameras see the road as if the car had drifted sideways.
    Labelling them with a corrected angle teaches the car how to recover."""
    paths = np.concatenate([df["center"], df["left"], df["right"]])
    s = df["steering"].to_numpy()
    angles = np.concatenate([s, s + correction, s - correction])
    return paths, np.clip(angles, -1.0, 1.0).astype(np.float32)


def batch_generator(images, angles, batch_size, rng):
    n = len(images)
    while True:
        idx = rng.integers(0, n, batch_size)
        x = np.empty((batch_size, 66, 200, 3), dtype=np.uint8)
        y = np.empty(batch_size, dtype=np.float32)
        for i, j in enumerate(idx):
            img, angle = augment(images[j], angles[j], rng)
            x[i] = preprocess(img)
            y[i] = angle
        yield x, y


def save_histograms(raw, balanced, train_angles, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, data, title in zip(
        axes,
        (raw, balanced, train_angles),
        ("Raw recording", "After balancing", "Training set (3 cameras)"),
    ):
        ax.hist(data, bins=25, range=(-1, 1), color="#3b6fb6")
        ax.set_title(title)
        ax.set_xlabel("steering angle")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def save_loss_plot(history, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["loss"], label="training loss")
    ax.plot(history["val_loss"], label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=["data/track"])
    p.add_argument("--out", default="models")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-per-bin", type=int, default=400)
    p.add_argument("--camera-correction", type=float, default=0.15)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    keras.utils.set_random_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    df = pd.concat([load_log(d) for d in args.data], ignore_index=True)
    raw_angles = df["steering"].to_numpy()
    df = balance(df, bins=25, max_per_bin=args.max_per_bin, rng=rng)
    print(f"Recorded frames: {len(raw_angles)}   after balancing: {len(df)}")

    # Split by frame, so no moment appears in both training and validation.
    order = rng.permutation(len(df))
    n_val = int(len(df) * args.val_split)
    val_df, train_df = df.iloc[order[:n_val]], df.iloc[order[n_val:]]

    train_paths, train_angles = expand_cameras(train_df, args.camera_correction)
    print(f"Training samples: {len(train_paths)} (center+left+right)   validation samples: {len(val_df)} (center)")
    save_histograms(raw_angles, df["steering"], train_angles, os.path.join(args.out, "steering_histogram.png"))

    print("Loading images into memory...")
    train_images = [load_rgb(p) for p in train_paths]
    x_val = np.stack([preprocess(load_rgb(p)) for p in val_df["center"]])
    y_val = val_df["steering"].to_numpy(dtype=np.float32)

    model = build_nvidia_model(args.lr)
    model.summary()

    best_path = os.path.join(args.out, "model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(args.out, "training_log.csv")),
    ]
    history = model.fit(
        batch_generator(train_images, train_angles, args.batch_size, rng),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=2,
    )

    model.save(best_path)  # best weights, restored by EarlyStopping
    model.save(os.path.join(args.out, "model.h5"))  # legacy format for older drive.py scripts
    save_loss_plot(history.history, os.path.join(args.out, "loss.png"))

    pred = model.predict(x_val, verbose=0).ravel()
    baseline = float(np.mean((y_val - y_val.mean()) ** 2))  # "always predict the average angle"
    summary = {
        "frames_recorded": int(len(raw_angles)),
        "frames_after_balancing": int(len(df)),
        "train_samples": int(len(train_paths)),
        "val_samples": int(len(val_df)),
        "epochs_run": len(history.history["loss"]),
        "best_val_mse": float(min(history.history["val_loss"])),
        "val_mae": float(np.mean(np.abs(pred - y_val))),
        "baseline_val_mse_constant_prediction": baseline,
        "args": vars(args),
    }
    with open(os.path.join(args.out, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
