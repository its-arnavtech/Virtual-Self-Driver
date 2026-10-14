"""NVIDIA's end-to-end self-driving network (a.k.a. PilotNet).

From "End to End Learning for Self-Driving Cars", Bojarski et al., NVIDIA 2016
(https://arxiv.org/abs/1604.07316). The paper describes it as 9 layers:

    1   normalization layer          pixels 0..255  ->  -1..1
    2-4 three 5x5 conv layers         stride 2: low-level features (edges, lane lines)
    5-6 two   3x3 conv layers         combine edges into shapes (road edges, curves)
    7-9 three fully connected layers  100 -> 50 -> 10: "boil it down" to a decision

followed by a single output neuron: the steering angle.

Input is a 66x200 image in YUV colour space, exactly as in the paper.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

INPUT_SHAPE = (66, 200, 3)  # height, width, channels (YUV)


def build_nvidia_model(learning_rate: float = 1e-4) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=INPUT_SHAPE),
            # Layer 1: normalization (done inside the network so drive.py can't forget it)
            layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalize"),
            # Layers 2-4: 5x5 convolutions with 2x2 stride -> edges and lines
            layers.Conv2D(24, 5, strides=2, activation="elu", name="conv1_edges"),
            layers.Conv2D(36, 5, strides=2, activation="elu", name="conv2_edges"),
            layers.Conv2D(48, 5, strides=2, activation="elu", name="conv3_edges"),
            # Layers 5-6: 3x3 convolutions, no stride -> shapes like road edges
            layers.Conv2D(64, 3, activation="elu", name="conv4_shapes"),
            layers.Conv2D(64, 3, activation="elu", name="conv5_shapes"),
            layers.Flatten(),  # 1x18x64 = 1152 numbers
            layers.Dropout(0.5),
            # Layers 7-9: fully connected -> narrow everything down to a decision
            layers.Dense(100, activation="elu", name="fc1"),
            layers.Dense(50, activation="elu", name="fc2"),
            layers.Dense(10, activation="elu", name="fc3"),
            # Output: one number, how far to turn the wheel (-1 full left .. +1 full right)
            layers.Dense(1, name="steering_angle"),
        ],
        name="nvidia_pilotnet",
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="mse")
    return model


if __name__ == "__main__":
    build_nvidia_model().summary()
