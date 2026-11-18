# isaac_ros_mcap_lerobot_converter

Converts MCAP rosbags directly into [LeRobot](https://github.com/huggingface/lerobot) datasets.
No ROS installation required — uses the pure-Python [`rosbags`](https://gitlab.com/ternaris/rosbags) library.

When a bag contains `isaac_ros_data_flywheel/msg/RecordData` messages, those
per-topic timestamps are used to synchronize joint states with camera frames.
Bags without `/record_data` fall back to nearest-neighbor timestamp matching
from the camera frames.

## Install

Requires **Python >= 3.10**; developed and tested against 3.12.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

Runtime dependencies (`av>=14.2,<19`, `lerobot==0.3.2`, `numpy`, `pillow`, `pyarrow`,
`pyyaml`, `rosbags`) are declared in `pyproject.toml` and are pulled in
automatically.

`package.xml` intentionally declares no runtime or test dependencies:
colcon will build the package but not install deps or run tests. The
dependencies and colcon test integration will be added back in a
follow-up MR.

## Usage

```bash
mcap-to-lerobot \
    --rosbags-dir /path/to/rosbags \
    --output-dir /path/to/lerobot_output \
    --task displayport_insertion \
    --fps 30 \
    --embodiment unitree_g1
```

Each subdirectory of `--rosbags-dir` that contains a `metadata.yaml` is
treated as one episode.  You can also point `--rosbags-dir` directly at a
single bag directory. Repeat `--rosbags-dir` to merge multiple recording
sessions into one dataset.

`--embodiment` selects both the dataset schema (joint layout, action
sources, teleop declarations and cameras) and the `robot_type` label written
to `info.json`. It must be one of the packaged embodiment names in
`isaac_ros_mcap_lerobot_converter/embodiments/`; an unknown value is
rejected up front with the list of valid names. The default is the existing
Unitree G1 conversion behavior (`--embodiment unitree_g1`). To export an H2
recording with Sharpa Wave hands, opt in explicitly:

```bash
mcap-to-lerobot \
    --rosbags-dir /path/to/h2_rosbags \
    --output-dir /path/to/lerobot_output \
    --task displayport_insertion \
    --fps 30 \
    --embodiment unitree_h2_sharpa_wave
```

The H2 embodiment merges `/applied_joint_commands` arm positions with causal
zero-order-held `/xr_teleop/finger_joints` positions into its 58-value action
vector. H2 camera and modality names come from the packaged embodiment YAML.

## Parallel conversion

Conversion runs entirely locally (no cluster/service dependency) and
converts bags in parallel across a process pool: each MCAP bag is decoded
and converted into a single-episode shard in its own worker process, and the
shards are then merged into the final dataset. Use `--num-workers` to
control how many bags are converted concurrently; it defaults to
`os.cpu_count()` (all cores on the machine). A bag that fails to convert is
logged and skipped rather than aborting the whole run, as long as at least
one bag succeeds.

```bash
mcap-to-lerobot \
    --rosbags-dir /path/to/rosbags \
    --output-dir /path/to/lerobot_output \
    --task displayport_insertion \
    --num-workers 8
```

## Optional GPU codecs

`--nvdec` and `--nvenc` control, independently, whether the NVIDIA
hardware video decoder (`h264_cuvid`, for reading camera footage out of the
bag) and encoder (`h264_nvenc`, for writing episode videos) are used. Both
accept `auto` (the default), `true`, or `false`:

- `auto` — use the hardware codec automatically when the environment's
  PyAV/ffmpeg build provides it, and silently fall back to software
  otherwise. This is the default and requires no extra setup.
- `true` — require the hardware codec; the converter errors out immediately
  (before opening any bag) if it is not available.
- `false` — always use software encode/decode, even if hardware support is
  present.

Neither `--nvdec` nor `--nvenc` is a hard dependency: the converter works
end-to-end on a machine with no NVIDIA GPU or codec support at all, using
`auto` (the default) or `false`.

```bash
mcap-to-lerobot \
    --rosbags-dir /path/to/rosbags \
    --output-dir /path/to/lerobot_output \
    --task displayport_insertion \
    --nvdec auto \
    --nvenc auto
```

## Running tests

```bash
uv pip install -e '.[test]'
pytest test/test_mcap_to_lerobot_converter.py -v
```

## Dependencies

| Package | Version |
|---------|---------|
| av      | >=14.2,<19 |
| lerobot | 0.3.2   |
| numpy   | unpinned |
| pillow  | unpinned |
| pyarrow | unpinned |
| pyyaml  | unpinned |
| rosbags | >= 0.10.11 |

## Supported image formats

Compressed camera topics (`sensor_msgs/CompressedImage`) must carry **H.264**
payloads — i.e. ``msg.format`` contains ``h264`` (or ``h.264``). JPEG and
PNG transports are rejected up front with a clear error. Raw
`sensor_msgs/Image` topics are supported in `rgb8`, `bgr8`, `rgba8`, and
`bgra8` encodings.
