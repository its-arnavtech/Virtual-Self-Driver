"""Download a public Udacity-simulator driving dataset into data/track.

    python scripts/download_data.py

Works on Windows, macOS and Linux (needs git on the PATH).

The dataset was recorded by a human driving in Udacity's open-source
self-driving-car simulator ("Training Mode"): 4,053 moments, each captured
by three dashboard cameras (center/left/right, 320x160 RGB) and logged with
the steering angle, throttle, brake and speed the driver used at that moment.
That is the exact format your own simulator recordings will have, so you can
later point train.py at your own data folder instead.

Source: https://github.com/rslim087a/track
"""

import os
import shutil
import stat
import subprocess
import sys

REPO = "https://github.com/rslim087a/track"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "track")


def _make_writable_and_retry(func, path, _exc):
    # git marks its object files read-only, which stops shutil.rmtree on Windows.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main():
    if os.path.isdir(os.path.join(DEST, "IMG")):
        print(f"{DEST} already exists, skipping download")
        return
    if shutil.which("git") is None:
        sys.exit("git is not installed or not on the PATH: https://git-scm.com/downloads")
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO, DEST], check=True)
    if sys.version_info >= (3, 12):
        shutil.rmtree(os.path.join(DEST, ".git"), onexc=_make_writable_and_retry)
    else:
        shutil.rmtree(os.path.join(DEST, ".git"), onerror=_make_writable_and_retry)

    images = len(os.listdir(os.path.join(DEST, "IMG")))
    with open(os.path.join(DEST, "driving_log.csv")) as f:
        rows = sum(1 for _ in f)
    print(f"Images: {images}   Log rows: {rows}")


if __name__ == "__main__":
    main()
