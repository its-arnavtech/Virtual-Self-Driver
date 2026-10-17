# Virtual-Self-Driver

A virtual car that learns to steer by itself using **NVIDIA's end-to-end self-driving network**
([Bojarski et al., 2016](https://arxiv.org/abs/1604.07316), the "PilotNet" model behind NVIDIA's
DAVE-2 car) and [Udacity's open-source self-driving car simulator](https://github.com/udacity/self-driving-car-sim).

To a computer a road isn't a road. It's 160 × 320 × 3 = 153,600 numbers per frame. The network
sees thousands of frames, each paired with the steering angle a human used at that moment, and
learns which patterns in those numbers mean "turn left", "turn right" or "go straight".

There are three steps: **drive, build, train.**

| Step | What happens | File |
|---|---|---|
| **Drive** (record) | A human drives in the simulator's *Training Mode*, which logs camera frames + steering angle | `scripts/download_data.py` fetches a ready-made recording |
| **Build** | The 9-layer NVIDIA network | `selfdriver/model.py` |
| **Train** | Show it the frames and angles until it can predict the angle itself | `train.py` |
| **Drive** (autonomous) | The trained network steers the car in the simulator's *Autonomous Mode* | `drive.py` |

## The network

```
input: 66x200 YUV image
 1  normalization                       pixel values 0..255 -> -1..1
 2  conv 5x5, 24 filters, stride 2  \
 3  conv 5x5, 36 filters, stride 2   |  low-level features: edges, lane lines
 4  conv 5x5, 48 filters, stride 2  /
 5  conv 3x3, 64 filters            \   combine edges into shapes:
 6  conv 3x3, 64 filters            /   road edges, curves
    flatten (1152 numbers) + dropout
 7  fully connected 100             \
 8  fully connected 50               |  boil it down to a decision
 9  fully connected 10              /
    output: 1 number = steering angle (-1 full left ... +1 full right)
```

252,219 trainable parameters, the same as in the paper. ELU activations, Adam optimizer, mean-squared-error loss.

## Dataset

The starter dataset is a public recording made in the Udacity simulator, mirrored at
[github.com/rslim087a/track](https://github.com/rslim087a/track):

* 4,053 moments of human driving on the lake track
* 3 dashboard cameras per moment (center / left / right), so 12,159 images of 320 × 160 px
* `driving_log.csv` holds `center, left, right, steering, throttle, brake, speed`

It uses exactly the format the simulator writes, so your own recordings work without changes.

How the data is prepared (`train.py`):

1. **Balance.** About 72% of the frames have a steering angle of exactly 0, because people mostly
   drive straight. Each of 25 steering ranges is capped (about 400 frames for the starter data;
   the cap grows with the amount of data), so the network can't get away with always going
   straight. See `models/steering_histogram.png`.
2. **Use all 3 cameras.** The left and right cameras see the road as if the car had drifted
   sideways. Their labels are adjusted by ±0.15, which teaches the car how to recover.
3. **Augment.** Each training image gets random zoom, pan, brightness changes and mirroring
   (with the angle mirrored too), so every batch looks a little different.
4. **Honest validation.** 20% of the data is held back to measure the network on road it hasn't
   trained on. Consecutive frames are nearly identical, so the split takes whole stretches of
   50 frames (about 3.5 s of driving) instead of single frames.
5. **Preprocess** (same in training and driving): crop out the sky and bonnet, convert to YUV,
   apply a slight blur, then resize to 66 × 200.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py     # ~350 MB into data/track (git-ignored), any OS
```

## Train

```bash
python train.py                                   # starter dataset
python train.py --data data/my_lap1 data/my_lap2  # your own simulator recordings (combined)
python train.py --help                            # epochs, learning rate, camera correction, ...
python train.py --resume models/model.keras --epochs 50  # keep training an existing model
```

Outputs go to `models/`:

* `model.keras` / `model.h5`: the trained network (the best epoch by validation loss)
* `loss.png`, `training_log.csv`: learning curves
* `steering_histogram.png`: steering-angle distribution before and after balancing
* `training_summary.json`: final metrics

Training runs fine on a CPU (about 1 minute per epoch on 4 cores). A GPU is faster.
Images stay JPEG-compressed in memory and are decoded on the fly, so several laps of your own
recordings fit comfortably in RAM.

## Tests

```bash
python -m pytest tests
```

## Record your own data (Drive)

1. Download the simulator from [udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim) (*Term 1*, version 2).
2. Choose **Training Mode**, press **R** to record, pick a folder (e.g. `data/my_lap1`), and drive.
   Several smooth laps in the center of the road work best. Add one lap in the opposite
   direction and a few "recover from the edge" clips.
3. Retrain: `python train.py --data data/track data/my_lap1`.

## Let it drive

```bash
python drive.py models/model.keras --speed 10
```

Open the simulator, choose **Autonomous Mode**, and the car steers itself.
