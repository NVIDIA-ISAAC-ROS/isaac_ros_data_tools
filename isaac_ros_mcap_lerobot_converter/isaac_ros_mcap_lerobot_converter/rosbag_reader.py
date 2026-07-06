# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python MCAP rosbag reader using the *rosbags* library.

No ROS installation is required.  Custom message types (e.g. RecordData)
are automatically registered from the schemas embedded in the MCAP file.
"""

import bisect
import io
import logging
from pathlib import Path

import av

import isaac_ros_mcap_lerobot_converter.structures as structures

import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
from rosbags.typesys.msg import get_types_from_msg

logger = logging.getLogger(__name__)

JOINT_STATE_MSGTYPE = 'sensor_msgs/msg/JointState'
JOINT_COMMAND_MSGTYPE = 'isaac_ros_deploy_interfaces/msg/JointCommand'
COMPRESSED_IMAGE_MSGTYPE = 'sensor_msgs/msg/CompressedImage'
RAW_IMAGE_MSGTYPE = 'sensor_msgs/msg/Image'
RECORD_DATA_MSGTYPE = 'isaac_ros_data_flywheel/msg/RecordData'
TWIST_STAMPED_MSGTYPE = 'geometry_msgs/msg/TwistStamped'
POSE_STAMPED_MSGTYPE = 'geometry_msgs/msg/PoseStamped'


class RosbagReader:
    """Reads an MCAP rosbag and extracts joint states, images, and sync records.

    Uses the ``isaac_ros_data_flywheel/msg/RecordData`` message for
    per-topic timestamp synchronization.  All data is kept in memory so
    callers can build a LeRobot dataset without writing an intermediate
    format to disk.
    """

    def __init__(self, bag_path: str, max_staleness_s: float | None = None):
        """Args:
            max_staleness_s: Max allowed age (in seconds) of a ZOH sample
                relative to the camera timestamp during synthetic sync.
                Frames where any required stream's latest prior sample is
                older than this are dropped. ``None`` disables the check.
        """
        self.bag_path = Path(bag_path)
        self.max_staleness_s = max_staleness_s

        self.robot_joint_states: list[structures.JointState] = []
        self.joint_commands: list[structures.JointState] = []
        self.compressed_camera_image_timestamps: dict[str, list[structures.Timestamp]] = {}
        self._image_buffers: dict[str, bytearray] = {}
        self._raw_image_frames: dict[str, list[np.ndarray]] = {}
        self._raw_camera_image_topics: list[str] = []
        self.record_data: list[structures.RecordData] = []
        self.compressed_camera_image_topics: list[str] = []
        self.navigate_commands: list[tuple[structures.Timestamp, list[float]]] = []
        self.base_height_commands: list[tuple[structures.Timestamp, list[float]]] = []

    def construct_dataset(self):
        """Read the entire bag and populate in-memory structures."""
        typestore = get_typestore(Stores.ROS2_JAZZY)

        with Reader(self.bag_path) as reader:
            self._register_custom_types(reader, typestore)
            self._discover_topics(reader)

            for connection, _timestamp, rawdata in reader.messages():
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                self._process_message(connection.topic, connection.msgtype, msg)

        # Generate synthetic sync records if the bag's RecordData is missing
        # or does not reference /applied_joint_commands stamps.
        has_cmd_stamps = any(
            ts.topic_name.strip('/') == 'applied_joint_commands'
            for rec in self.record_data for ts in rec.topic_stamps
        )
        if not has_cmd_stamps and self.robot_joint_states and self.joint_commands:
            if self.record_data:
                logger.warning(
                    'Discarding %d /record_data entries: none reference '
                    '/applied_joint_commands (likely recorder config lag). '
                    'Falling back to synthetic causal ZOH sync.',
                    len(self.record_data),
                )
            self.record_data = []
            self._generate_synthetic_record_data(
                max_staleness_s=self.max_staleness_s,
            )

        logger.info(
            'Read %d joint states, %d joint commands, %d record entries, '
            '%d camera topics, %d nav cmds, %d height cmds',
            len(self.robot_joint_states),
            len(self.joint_commands),
            len(self.record_data),
            len(self.compressed_camera_image_topics),
            len(self.navigate_commands),
            len(self.base_height_commands),
        )

    @staticmethod
    def _register_custom_types(reader: Reader, typestore) -> None:
        """Register any non-standard message types found in the bag schemas."""
        add_types: dict = {}
        for connection in reader.connections:
            if not connection.msgdef:
                continue
            msgdef = connection.msgdef
            msgdef_text = getattr(msgdef, 'data', msgdef)
            if not isinstance(msgdef_text, str) or not msgdef_text.strip():
                continue
            try:
                add_types.update(
                    get_types_from_msg(msgdef_text, connection.msgtype)
                )
            except Exception as e:
                logger.debug('Skipping schema for %s: %s',
                             connection.msgtype, e)
        if add_types:
            typestore.register(add_types)

    def _discover_topics(self, reader: Reader) -> None:
        """Inspect bag connections to identify relevant topics."""
        self._robot_joint_state_topic: str | None = None
        self._joint_commands_topic: str | None = None
        self._record_data_topic: str | None = None
        self._root_twist_topic: str | None = None
        self._root_pose_topic: str | None = None

        joint_state_candidates: list[tuple[str, str]] = []
        joint_command_candidates: list[tuple[str, str]] = []

        for conn in reader.connections:
            if conn.msgtype == JOINT_STATE_MSGTYPE:
                joint_state_candidates.append((conn.topic, conn.msgtype))
            if conn.msgtype == JOINT_COMMAND_MSGTYPE:
                joint_command_candidates.append((conn.topic, conn.msgtype))

            if conn.msgtype == JOINT_STATE_MSGTYPE and conn.topic == '/joint_states':
                self._robot_joint_state_topic = conn.topic
            elif conn.msgtype == JOINT_COMMAND_MSGTYPE and conn.topic == '/applied_joint_commands':
                self._joint_commands_topic = conn.topic
            elif conn.msgtype == COMPRESSED_IMAGE_MSGTYPE:
                if 'compressedDepth' not in conn.topic:
                    self.compressed_camera_image_topics.append(conn.topic)
            elif conn.msgtype == RAW_IMAGE_MSGTYPE:
                if 'depth' not in conn.topic:
                    self._raw_camera_image_topics.append(conn.topic)
                    self.compressed_camera_image_topics.append(conn.topic)
            elif conn.msgtype == RECORD_DATA_MSGTYPE:
                self._record_data_topic = conn.topic
            elif (conn.msgtype == TWIST_STAMPED_MSGTYPE
                  and 'root_twist' in conn.topic):
                self._root_twist_topic = conn.topic
            elif (conn.msgtype == POSE_STAMPED_MSGTYPE
                  and 'root_pose' in conn.topic):
                self._root_pose_topic = conn.topic

        if self._robot_joint_state_topic is None:
            raise ValueError(
                f'Topic /joint_states ({JOINT_STATE_MSGTYPE}) not found in bag. '
                f'JointState topics present: {joint_state_candidates or "none"}'
            )
        if self._joint_commands_topic is None:
            raise ValueError(
                f'Topic /applied_joint_commands ({JOINT_COMMAND_MSGTYPE}) not '
                f'found in bag. JointCommand topics present: '
                f'{joint_command_candidates or "none"}'
            )
        if self._record_data_topic is None:
            logger.warning('RecordData topic not found — will generate '
                           'synthetic sync from timestamps')

    def resync_at_rate(self, target_rate_hz: float) -> None:
        """Discard the bag's ``/record_data`` and rebuild it via causal ZOH at
        ``target_rate_hz``.  Call after ``construct_dataset()`` when the
        caller wants the dataset row count to be driven by a different rate
        than the bag was recorded at.
        """
        self.record_data = []
        self._generate_synthetic_record_data(
            max_staleness_s=self.max_staleness_s,
            target_rate_hz=target_rate_hz,
        )

    def _generate_synthetic_record_data(
        self,
        max_staleness_s: float | None = None,
        target_rate_hz: float | None = None,
    ):
        """Generate RecordData entries via causal zero-order hold (ZOH).

        For each tick (see below), pick the most recent sample on each
        topic with timestamp <= the tick.  Ticks for which any required
        stream has no prior sample, or whose latest prior sample is older
        than ``max_staleness_s``, are dropped.

        Tick source:

        * ``target_rate_hz=None`` (default) — one tick per camera frame.
          This is the original fallback used when the bag is missing
          ``/record_data``: dataset rate equals the camera native rate.
        * ``target_rate_hz=<hz>`` — fixed-period ticks at ``1/hz`` spanning
          the camera timeline.  Used by the converter when the user asks
          for a dataset rate that differs from the bag's recording rate
          (the bag's existing ``/record_data`` is discarded by the caller
          before this is called).
        """
        # bisect_right returns the insertion point after all entries <= target,
        # so stamps[idx-1] is the rightmost sample with ts <= target. Guard
        # idx == 0 (no prior sample exists) by returning None.
        staleness_ns = (
            int(max_staleness_s * 1e9) if max_staleness_s is not None else None
        )

        def _ts_ns(ts):
            return ts.seconds * 10**9 + ts.nanoseconds

        def _sorted_ns(stamps):
            """Return (stamps, stamps_ns) sorted by ns for bisect lookup."""
            indexed = sorted(stamps, key=_ts_ns)
            return indexed, [_ts_ns(s) for s in indexed]

        def _zoh(stamps, stamps_ns, target_ns):
            idx = bisect.bisect_right(stamps_ns, target_ns)
            if idx == 0:
                return None
            if staleness_ns is not None and target_ns - stamps_ns[idx - 1] > staleness_ns:
                return None
            return stamps[idx - 1]

        js_stamps, js_ns = _sorted_ns(
            [js.timestamp for js in self.robot_joint_states])
        cmd_stamps, cmd_ns = _sorted_ns(
            [js.timestamp for js in self.joint_commands])
        nav_stamps, nav_ns = _sorted_ns(
            [ts for ts, _ in self.navigate_commands])
        height_stamps, height_ns = _sorted_ns(
            [ts for ts, _ in self.base_height_commands])
        cam_indexed = {}
        for topic in self.compressed_camera_image_topics:
            stamps = self.compressed_camera_image_timestamps.get(topic, [])
            cam_indexed[topic] = _sorted_ns(stamps)

        all_cam = []
        for stamps, _ in cam_indexed.values():
            all_cam.extend(stamps)
        if not all_cam:
            return

        if target_rate_hz is None:
            tick_ns_iter = sorted(_ts_ns(c) for c in all_cam)
        else:
            if target_rate_hz <= 0:
                raise ValueError(
                    f'target_rate_hz must be positive, got {target_rate_hz}')
            sorted_cam_ns = sorted(_ts_ns(c) for c in all_cam)
            t_start, t_end = sorted_cam_ns[0], sorted_cam_ns[-1]
            period_ns = int(1e9 / target_rate_hz)
            tick_ns_iter = range(t_start, t_end + 1, period_ns)

        for tick_ns in tick_ns_iter:
            cam_ns = tick_ns
            js = _zoh(js_stamps, js_ns, cam_ns)
            cmd = _zoh(cmd_stamps, cmd_ns, cam_ns)
            if js is None or cmd is None:
                continue

            topic_stamps = [
                structures.TopicStamp(topic_name='joint_states',
                                      stamp=js, delay_ms=0.0),
                structures.TopicStamp(topic_name='applied_joint_commands',
                                      stamp=cmd, delay_ms=0.0),
            ]
            for topic, (stamps, stamps_ns) in cam_indexed.items():
                cam = _zoh(stamps, stamps_ns, cam_ns)
                if cam:
                    topic_stamps.append(structures.TopicStamp(
                        topic_name=topic, stamp=cam, delay_ms=0.0))
            if nav_stamps:
                nav = _zoh(nav_stamps, nav_ns, cam_ns)
                if nav:
                    topic_stamps.append(structures.TopicStamp(
                        topic_name='xr_teleop/root_twist', stamp=nav,
                        delay_ms=0.0))
            if height_stamps:
                height = _zoh(height_stamps, height_ns, cam_ns)
                if height:
                    topic_stamps.append(structures.TopicStamp(
                        topic_name='xr_teleop/root_pose', stamp=height,
                        delay_ms=0.0))
            self.record_data.append(
                structures.RecordData(topic_stamps=topic_stamps))

        if target_rate_hz is None:
            logger.info('Generated %d synthetic sync records (camera-locked)',
                        len(self.record_data))
        else:
            logger.info(
                'Generated %d synthetic sync records at %.2f Hz',
                len(self.record_data), target_rate_hz)

    def _process_message(self, topic: str, _msgtype: str, msg) -> None:
        if topic == self._robot_joint_state_topic:
            self.robot_joint_states.append(self._parse_joint_state(msg))

        elif topic == self._joint_commands_topic:
            self.joint_commands.append(self._parse_joint_state(msg))

        elif topic in self.compressed_camera_image_topics:
            ts = structures.Timestamp(
                seconds=msg.header.stamp.sec,
                nanoseconds=msg.header.stamp.nanosec,
            )
            if topic not in self.compressed_camera_image_timestamps:
                self.compressed_camera_image_timestamps[topic] = []
            self.compressed_camera_image_timestamps[topic].append(ts)

            if topic in self._raw_camera_image_topics:
                if topic not in self._raw_image_frames:
                    self._raw_image_frames[topic] = []
                self._raw_image_frames[topic].append(
                    self._parse_raw_image(msg)
                )
            else:
                # CompressedImage.format is a free-form string like
                # 'h264' or 'bgr8; jpeg compressed bgr8'. decode_video_frames
                # hands the buffer to PyAV with format='h264'; anything else
                # would silently produce garbage or crash the decoder, so
                # fail fast here with the topic and reported format instead.
                fmt = getattr(msg, 'format', '') or ''
                if 'h264' not in fmt.lower() and 'h.264' not in fmt.lower():
                    raise ValueError(
                        f"Compressed image on {topic!r} has format {fmt!r} "
                        f"but this converter only supports H.264. See README.")
                if topic not in self._image_buffers:
                    self._image_buffers[topic] = bytearray()
                self._image_buffers[topic].extend(bytes(msg.data))

        elif topic == self._record_data_topic:
            self._parse_record_data(msg)

        elif topic == self._root_twist_topic:
            ts = structures.Timestamp(
                seconds=msg.header.stamp.sec,
                nanoseconds=msg.header.stamp.nanosec,
            )
            self.navigate_commands.append(
                (ts, [float(msg.twist.linear.x), float(msg.twist.linear.y),
                      float(msg.twist.angular.z)])
            )

        elif topic == self._root_pose_topic:
            ts = structures.Timestamp(
                seconds=msg.header.stamp.sec,
                nanoseconds=msg.header.stamp.nanosec,
            )
            self.base_height_commands.append(
                (ts, [float(msg.pose.position.z)])
            )

    @staticmethod
    def _parse_joint_state(msg) -> structures.JointState:
        # sensor_msgs/JointState spells it `name`, isaac_ros_deploy_interfaces
        # /JointCommand spells it `names` — accept either.
        names = getattr(msg, 'name', None)
        if names is None:
            names = msg.names
        return structures.JointState(
            position=list(msg.position),
            velocity=list(msg.velocity),
            effort=list(msg.effort),
            names=list(names),
            timestamp=structures.Timestamp(
                seconds=msg.header.stamp.sec,
                nanoseconds=msg.header.stamp.nanosec,
            ),
        )

    def _parse_record_data(self, msg) -> None:
        self.record_data.append(structures.RecordData(
            topic_stamps=[
                structures.TopicStamp(
                    topic_name=ts.topic_name,
                    stamp=structures.Timestamp(
                        seconds=ts.stamp.sec,
                        nanoseconds=ts.stamp.nanosec,
                    ),
                    delay_ms=ts.delay_ms,
                )
                for ts in msg.topic_stamps
            ],
        ))

    @staticmethod
    def _parse_raw_image(msg) -> np.ndarray:
        """Convert a sensor_msgs/msg/Image message to an RGB numpy array."""
        h, w = msg.height, msg.width
        encoding = msg.encoding
        data = bytes(msg.data)
        if encoding == 'rgb8':
            return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        elif encoding == 'bgr8':
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            return frame[:, :, ::-1].copy()
        elif encoding == 'rgba8':
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
            return frame[:, :, :3].copy()
        elif encoding == 'bgra8':
            frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
            return frame[:, :, 2::-1].copy()
        else:
            raise ValueError(f'Unsupported image encoding: {encoding}')

    def decode_video_frames(self, topic: str) -> list[np.ndarray]:
        """Decode frames for *topic* into a list of RGB numpy arrays.

        ``container.decode`` internally flushes the decoder's B-frame
        reorder buffer at end-of-stream, so raw Annex-B H264 streams
        parsed from memory recover every frame. See
        ``test_decoded_frame_count_matches_messages`` for the invariant.
        """
        if topic in self._raw_image_frames:
            return self._raw_image_frames[topic]

        if topic not in self._image_buffers or not self._image_buffers[topic]:
            return []

        buffer = io.BytesIO(bytes(self._image_buffers[topic]))
        with av.open(buffer, format='h264') as container:
            return [
                frame.to_ndarray(format='rgb24')
                for frame in container.decode(video=0)
            ]
