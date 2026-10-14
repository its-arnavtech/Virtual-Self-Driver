"""Step 1/finale: DRIVE. Let the trained network steer the car in the Udacity simulator.

    python drive.py                      # loads models/model.keras
    python drive.py models/model.h5 --speed 12

Then open the simulator, pick a track and click "Autonomous Mode". The simulator
connects to this server on port 4567, sends a camera frame plus the current
speed many times a second, and we answer with a steering angle and throttle.

The simulator speaks the old Socket.IO v2 protocol, which is why
requirements.txt pins python-socketio 4.x / python-engineio 3.x.
"""

import argparse
import base64
from io import BytesIO

import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow import keras

from selfdriver.preprocess import preprocess

sio = socketio.Server()
app = Flask(__name__)
model = None
speed_limit = 10.0


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:  # simulator is in manual mode
        sio.emit("manual", data={}, skip_sid=True)
        return
    speed = float(data["speed"])
    frame = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB"))
    x = preprocess(frame)[None, ...]
    steering = float(model(x, training=False)[0, 0])
    throttle = 1.0 - speed / speed_limit  # simple speed controller: ease off near the limit
    print(f"steering {steering:+.3f}   throttle {throttle:+.3f}   speed {speed:5.2f}")
    send_control(steering, throttle)


@sio.on("connect")
def connect(sid, environ):
    print("Simulator connected")
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={"steering_angle": str(steering_angle), "throttle": str(throttle)})


def main():
    global model, speed_limit
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", default="models/model.keras")
    p.add_argument("--speed", type=float, default=10.0, help="target speed in mph")
    p.add_argument("--port", type=int, default=4567)
    args = p.parse_args()

    speed_limit = args.speed
    model = keras.models.load_model(args.model)
    wsgi_app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", args.port)), wsgi_app)


if __name__ == "__main__":
    main()
