#!/usr/bin/env bash
# Downloads a public Udacity-simulator driving dataset into data/track.
#
# The dataset was recorded by a human driving in Udacity's open-source
# self-driving-car simulator ("Training Mode"): 4,053 moments, each captured
# by three dashboard cameras (center/left/right, 320x160 RGB) and logged with
# the steering angle, throttle, brake and speed the driver used at that moment.
# That is the exact format your own simulator recordings will have, so you can
# later point train.py at your own data folder instead.
#
# Source: https://github.com/rslim087a/track
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ -d data/track/IMG ]; then
  echo "data/track already exists, skipping download"
  exit 0
fi
git clone --depth 1 https://github.com/rslim087a/track data/track
rm -rf data/track/.git
echo "Images: $(ls data/track/IMG | wc -l)   Log rows: $(wc -l < data/track/driving_log.csv)"
