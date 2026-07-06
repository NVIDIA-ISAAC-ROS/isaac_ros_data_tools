# isaac_ros_mcap_lerobot_converter

Converts MCAP rosbags directly into [LeRobot](https://github.com/huggingface/lerobot) datasets.
No ROS installation required — uses the pure-Python [`rosbags`](https://gitlab.com/ternaris/rosbags) library.

When a bag contains `isaac_ros_data_flywheel/msg/RecordData` messages, those
per-topic timestamps are used to synchronize joint states with camera frames.
Bags without `/record_data` fall back to nearest-neighbor timestamp matching
from the camera frames.

## Install

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

Runtime dependencies (`lerobot==0.3.2`, `rosbags`, `av`, `numpy`) are
declared in `pyproject.toml` and are pulled in automatically.

`package.xml` intentionally declares no runtime or test dependencies:
colcon will build the package but not install deps or run tests. The
dependencies and colcon test integration will be added back in a
follow-up MR.

## Usage

```bash
mcap-to-lerobot \
    --rosbags-dir /path/to/rosbags \
    --output-dir /path/to/lerobot_output \
    --task gear_insertion \
    --fps 30 \
    --robot-type ur10e
```

Each subdirectory of `--rosbags-dir` that contains a `metadata.yaml` is
treated as one episode.  You can also point `--rosbags-dir` directly at a
single bag directory.

## Running tests

```bash
uv pip install pytest
pytest test/test_mcap_to_lerobot_converter.py -v
```

## Pinned dependencies

| Package | Version |
|---------|---------|
| lerobot | 0.3.2   |
| rosbags | >= 0.10.11 |

## Supported image formats

Compressed camera topics (`sensor_msgs/CompressedImage`) must carry **H.264**
payloads — i.e. ``msg.format`` contains ``h264`` (or ``h.264``). JPEG and
PNG transports are rejected up front with a clear error. Raw
`sensor_msgs/Image` topics are supported in `rgb8`, `bgr8`, `rgba8`, and
`bgra8` encodings.
