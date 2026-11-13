#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 Emotion Control Node for TurtleBot 4
Node: emotion_control_node

Subscribes:
  - /detected_emotion (std_msgs/msg/String): Canonical emotion label
    ('Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Fear').
  - /odom (nav_msgs/msg/Odometry): Odometry yaw integration for 360-deg Surprise spin.
  - /scan (sensor_msgs/msg/LaserScan): LiDAR obstacle distances (front & rear).

Publishes:
  - /cmd_vel (geometry_msgs/msg/Twist) at 20 Hz (0.05s timer).
  - /emotion_robot_status (std_msgs/msg/String): Current state/action summary for HUD/logs.

Service Client:
  - /motion_control/set_parameters (rcl_interfaces/srv/SetParameters):
    Automatically sets Create 3 `safety_override='full'` so negative linear.x
    (reverse motion for Angry) works in Gazebo and on TurtleBot 4 hardware.
"""

import math
import time
from typing import Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue


CANONICAL_EMOTIONS = ('Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Fear')

EMOTION_ALIASES = {
    'happy': 'Happy',
    'vui': 'Happy',
    'vui ve': 'Happy',
    'vui_ve': 'Happy',
    'vui vẻ': 'Happy',
    'sad': 'Sad',
    'buon': 'Sad',
    'buồn': 'Sad',
    'neutral': 'Neutral',
    'binh thuong': 'Neutral',
    'binh_thuong': 'Neutral',
    'bình thường': 'Neutral',
    'disgust': 'Neutral',
    'none': 'Neutral',
    'stop': 'Neutral',
    '': 'Neutral',
    'angry': 'Angry',
    'anger': 'Angry',
    'tuc gian': 'Angry',
    'tuc_gian': 'Angry',
    'tức giận': 'Angry',
    'surprise': 'Surprise',
    'surprised': 'Surprise',
    'ngac nhien': 'Surprise',
    'ngac_nhien': 'Surprise',
    'ngạc nhiên': 'Surprise',
    'fear': 'Fear',
    'fearful': 'Fear',
    'scared': 'Fear',
    'so hai': 'Fear',
    'so_hai': 'Fear',
    'sợ hãi': 'Fear',
}


def normalize_emotion(raw_emotion: Optional[str]) -> str:
    """Normalize an incoming emotion label to one of the 6 canonical emotions."""
    if not isinstance(raw_emotion, str):
        return 'Neutral'
    cleaned = raw_emotion.strip()
    if cleaned in CANONICAL_EMOTIONS:
        return cleaned
    normalized_key = ' '.join(cleaned.lower().replace('-', '_').split())
    return EMOTION_ALIASES.get(normalized_key, 'Neutral')


def normalize_angle(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert a geometry_msgs quaternion (x, y, z, w) into planar yaw (rad)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_twist_for_emotion(
    emotion: str,
    curve_linear_speed: float = 0.18,
    curve_angular_speed: float = 0.55,
    gentle_linear_speed: float = 0.12,
    spin_angular_speed: float = 1.0,
    surprise_active: bool = True,
) -> Tuple[float, float]:
    """
    Compute (linear_x, angular_z) velocity command for a given emotion.

    Mapping:
      - Happy    : Curve left  (+curve_linear_speed, +curve_angular_speed)
      - Sad      : Curve right (+curve_linear_speed, -curve_angular_speed)
      - Neutral  : Stop        (0.0, 0.0)
      - Angry    : Back up     (-gentle_linear_speed, 0.0)
      - Surprise : Spin 360    (0.0, +spin_angular_speed) when surprise_active else (0.0, 0.0)
      - Fear     : Move fwd    (+gentle_linear_speed, 0.0)
    """
    canonical = normalize_emotion(emotion)
    if canonical == 'Happy':
        return float(curve_linear_speed), float(curve_angular_speed)
    if canonical == 'Sad':
        return float(curve_linear_speed), -float(curve_angular_speed)
    if canonical == 'Angry':
        return -float(gentle_linear_speed), 0.0
    if canonical == 'Surprise':
        if surprise_active:
            return 0.0, float(spin_angular_speed)
        return 0.0, 0.0
    if canonical == 'Fear':
        return float(gentle_linear_speed), 0.0
    return 0.0, 0.0


def compute_scan_clearances(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: Optional[float] = None,
    front_half_angle_deg: float = 30.0,
    rear_half_angle_deg: float = 30.0,
) -> Tuple[float, float]:
    """Compute minimum valid obstacle distance (m) in front and rear LiDAR sectors."""
    if not ranges:
        return float('inf'), float('inf')

    front_rad = math.radians(front_half_angle_deg)
    rear_rad = math.radians(180.0 - rear_half_angle_deg)
    front_min = float('inf')
    rear_min = float('inf')

    angle = angle_min
    for r in ranges:
        if math.isfinite(r) and r > range_min and (range_max is None or range_max <= 0.0 or r <= range_max):
            norm_ang = abs(normalize_angle(angle))
            if norm_ang <= front_rad and r < front_min:
                front_min = float(r)
            elif norm_ang >= rear_rad and r < rear_min:
                rear_min = float(r)
        angle += angle_increment

    return front_min, rear_min


def apply_lidar_safety(
    linear_x: float,
    angular_z: float,
    front_clear_dist: float,
    rear_clear_dist: float,
    obstacle_stop_dist: float = 0.25,
    enable_lidar_safety: bool = True,
) -> Tuple[float, float, Optional[str]]:
    """
    Gate forward/reverse motion when LiDAR clearance falls below obstacle_stop_dist.

    Returns (safe_linear_x, safe_angular_z, blocked_side), where blocked_side is
    'front', 'rear', or None.
    """
    if not enable_lidar_safety:
        return float(linear_x), float(angular_z), None

    if linear_x > 0.0 and front_clear_dist < obstacle_stop_dist:
        return 0.0, 0.0, 'front'
    if linear_x < 0.0 and rear_clear_dist < obstacle_stop_dist:
        return 0.0, 0.0, 'rear'
    return float(linear_x), float(angular_z), None


class EmotionMotionController:
    """
    Pure-Python emotion-to-velocity state machine for TurtleBot 4.

    Handles:
      - Watchdog timeout when /detected_emotion stops arriving
      - Surprise 360-degree spin with /odom yaw integration + duration fallback
      - Edge-trigger/latch for Surprise (one 360-deg spin per Surprise hold)
      - Immediate cancellation of Surprise spin on Neutral or any other emotion
      - Front and rear LiDAR safety gating
    """

    def __init__(
        self,
        curve_linear_speed: float = 0.18,
        curve_angular_speed: float = 0.55,
        gentle_linear_speed: float = 0.12,
        spin_angular_speed: float = 1.0,
        signal_timeout_sec: float = 1.5,
        enable_lidar_safety: bool = True,
        obstacle_stop_dist: float = 0.25,
        surprise_target_angle_rad: float = 2.0 * math.pi,
        surprise_cooldown_sec: float = 0.0,
    ):
        self.curve_linear_speed = float(curve_linear_speed)
        self.curve_angular_speed = float(curve_angular_speed)
        self.gentle_linear_speed = float(gentle_linear_speed)
        self.spin_angular_speed = float(spin_angular_speed)
        self.signal_timeout_sec = float(signal_timeout_sec)
        self.enable_lidar_safety = bool(enable_lidar_safety)
        self.obstacle_stop_dist = float(obstacle_stop_dist)
        self.surprise_target_angle_rad = float(surprise_target_angle_rad)
        self.surprise_cooldown_sec = float(surprise_cooldown_sec)

        self.current_emotion = 'Neutral'
        self.last_msg_time: Optional[float] = None
        self.watchdog_timed_out = False

        # LiDAR clearances
        self.front_clear_dist = float('inf')
        self.rear_clear_dist = float('inf')

        # Odometry & Surprise 360-degree spin state
        self.has_odom = False
        self.robot_yaw = 0.0
        self.surprise_spinning = False
        self.surprise_latched = False
        self.surprise_start_time = 0.0
        self.surprise_completed_time: Optional[float] = None
        self.surprise_accumulated_yaw = 0.0
        self.surprise_last_odom_yaw: Optional[float] = None
        self.surprise_odom_updates = 0

    def _start_surprise_spin(self, now: float) -> None:
        self.surprise_spinning = True
        self.surprise_latched = True
        self.surprise_start_time = now
        self.surprise_completed_time = None
        self.surprise_accumulated_yaw = 0.0
        self.surprise_last_odom_yaw = self.robot_yaw if self.has_odom else None
        self.surprise_odom_updates = 0

    def _finish_surprise_spin(self, now: float) -> None:
        self.surprise_spinning = False
        self.surprise_completed_time = now

    def _fallback_spin_duration(self) -> float:
        return self.surprise_target_angle_rad / max(abs(self.spin_angular_speed), 1e-3)

    def on_emotion(self, raw_emotion: str, now: float) -> str:
        """Process a newly received emotion string at timestamp `now`."""
        emotion = normalize_emotion(raw_emotion)
        self.last_msg_time = now
        self.watchdog_timed_out = False

        if emotion != 'Surprise':
            # Changing away from Surprise (including Neutral) cancels any active spin
            # and clears the latch so a subsequent Surprise triggers a new 360 spin.
            self.surprise_spinning = False
            self.surprise_latched = False
            self.current_emotion = emotion
            return emotion

        self.current_emotion = 'Surprise'
        if not self.surprise_latched:
            self._start_surprise_spin(now)
        elif (
            not self.surprise_spinning
            and self.surprise_cooldown_sec > 0.0
            and self.surprise_completed_time is not None
            and (now - self.surprise_completed_time) >= self.surprise_cooldown_sec
        ):
            self._start_surprise_spin(now)
        return emotion

    def on_odom(self, yaw: float, now: Optional[float] = None) -> None:
        """Update planar yaw from /odom and integrate Surprise spin angle when active."""
        self.has_odom = True
        self.robot_yaw = float(yaw)
        if not self.surprise_spinning:
            return

        if self.surprise_last_odom_yaw is None:
            self.surprise_last_odom_yaw = self.robot_yaw
            return

        delta = normalize_angle(self.robot_yaw - self.surprise_last_odom_yaw)
        spin_dir = 1.0 if self.spin_angular_speed >= 0.0 else -1.0
        self.surprise_accumulated_yaw += spin_dir * delta
        self.surprise_last_odom_yaw = self.robot_yaw
        self.surprise_odom_updates += 1

        if self.surprise_accumulated_yaw >= self.surprise_target_angle_rad:
            self._finish_surprise_spin(time.monotonic() if now is None else now)

    def on_scan(self, front_clear_dist: float, rear_clear_dist: float) -> None:
        """Update front and rear obstacle clearance distances from /scan."""
        self.front_clear_dist = float(front_clear_dist)
        self.rear_clear_dist = float(rear_clear_dist)

    def step(self, now: float) -> Tuple[float, float, str]:
        """Compute the current (linear_x, angular_z, status_text) command at time `now`."""
        if self.last_msg_time is None:
            return 0.0, 0.0, 'WAITING: Cho tin hieu /detected_emotion...'

        if (now - self.last_msg_time) > self.signal_timeout_sec:
            self.watchdog_timed_out = True
            self.surprise_spinning = False
            self.surprise_latched = False
            return (
                0.0,
                0.0,
                f'STOPPED (Watchdog): Mat tin hieu /detected_emotion > {self.signal_timeout_sec:.1f}s',
            )

        if self.current_emotion == 'Surprise' and self.surprise_spinning:
            if self.surprise_odom_updates > 0:
                if self.surprise_accumulated_yaw >= self.surprise_target_angle_rad:
                    self._finish_surprise_spin(now)
            else:
                if (now - self.surprise_start_time) >= self._fallback_spin_duration():
                    self._finish_surprise_spin(now)

        raw_lin_x, raw_ang_z = compute_twist_for_emotion(
            self.current_emotion,
            curve_linear_speed=self.curve_linear_speed,
            curve_angular_speed=self.curve_angular_speed,
            gentle_linear_speed=self.gentle_linear_speed,
            spin_angular_speed=self.spin_angular_speed,
            surprise_active=self.surprise_spinning,
        )

        safe_lin_x, safe_ang_z, blocked_side = apply_lidar_safety(
            raw_lin_x,
            raw_ang_z,
            self.front_clear_dist,
            self.rear_clear_dist,
            obstacle_stop_dist=self.obstacle_stop_dist,
            enable_lidar_safety=self.enable_lidar_safety,
        )

        if blocked_side == 'front':
            status = (
                f'STOPPED (Front LiDAR): Vat can phia truoc '
                f'{self.front_clear_dist:.2f}m < {self.obstacle_stop_dist:.2f}m [{self.current_emotion}]'
            )
            return safe_lin_x, safe_ang_z, status

        if blocked_side == 'rear':
            status = (
                f'STOPPED (Rear LiDAR): Vat can phia sau '
                f'{self.rear_clear_dist:.2f}m < {self.obstacle_stop_dist:.2f}m [{self.current_emotion}]'
            )
            return safe_lin_x, safe_ang_z, status

        status = self._format_active_status(now, safe_lin_x, safe_ang_z)
        return safe_lin_x, safe_ang_z, status

    def _format_active_status(self, now: float, lin_x: float, ang_z: float) -> str:
        if self.current_emotion == 'Happy':
            return f'MOVING [Happy - Vui ve]: Re trai cong (v={lin_x:+.2f}, w={ang_z:+.2f})'
        if self.current_emotion == 'Sad':
            return f'MOVING [Sad - Buon]: Re phai cong (v={lin_x:+.2f}, w={ang_z:+.2f})'
        if self.current_emotion == 'Angry':
            return f'MOVING [Angry - Tuc gian]: Lui nhe (v={lin_x:+.2f}, w={ang_z:+.2f})'
        if self.current_emotion == 'Fear':
            return f'MOVING [Fear - So hai]: Tien nhe (v={lin_x:+.2f}, w={ang_z:+.2f})'
        if self.current_emotion == 'Surprise':
            if self.surprise_spinning:
                if self.surprise_odom_updates > 0:
                    deg = math.degrees(max(0.0, self.surprise_accumulated_yaw))
                    return f'EXECUTING [Surprise - Ngac nhien]: Xoay 360 ({deg:.0f}/360 deg)'
                elapsed = max(0.0, now - self.surprise_start_time)
                return (
                    f'EXECUTING [Surprise - Ngac nhien]: Xoay 360 '
                    f'({elapsed:.1f}/{self._fallback_spin_duration():.1f}s)'
                )
            return 'COMPLETED [Surprise - Ngac nhien]: Da xoay du 360 deg (latched)'
        return 'STOPPED [Neutral - Binh thuong]: Dung tai cho'


class EmotionControlNode(Node):
    """ROS 2 Node translating /detected_emotion into safe /cmd_vel trajectories."""

    def __init__(self):
        super().__init__('emotion_control_node')

        # Trajectory, safety, and timing parameters
        self.declare_parameter('curve_linear_speed', 0.18)
        self.declare_parameter('curve_angular_speed', 0.55)
        self.declare_parameter('gentle_linear_speed', 0.12)
        self.declare_parameter('spin_angular_speed', 1.0)
        self.declare_parameter('signal_timeout_sec', 1.5)
        self.declare_parameter('enable_lidar_safety', True)
        self.declare_parameter('obstacle_stop_dist', 0.25)
        self.declare_parameter('surprise_target_angle_rad', 2.0 * math.pi)
        self.declare_parameter('surprise_cooldown_sec', 0.0)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('safety_override_value', 'full')

        curve_v = float(self.get_parameter('curve_linear_speed').value)
        curve_w = float(self.get_parameter('curve_angular_speed').value)
        gentle_v = float(self.get_parameter('gentle_linear_speed').value)
        spin_w = float(self.get_parameter('spin_angular_speed').value)
        signal_timeout = float(self.get_parameter('signal_timeout_sec').value)
        enable_lidar = bool(self.get_parameter('enable_lidar_safety').value)
        stop_dist = float(self.get_parameter('obstacle_stop_dist').value)
        surprise_angle = float(self.get_parameter('surprise_target_angle_rad').value)
        surprise_cooldown = float(self.get_parameter('surprise_cooldown_sec').value)
        control_rate_hz = max(1.0, float(self.get_parameter('control_rate_hz').value))
        self.safety_override_value = str(self.get_parameter('safety_override_value').value)

        self.controller = EmotionMotionController(
            curve_linear_speed=curve_v,
            curve_angular_speed=curve_w,
            gentle_linear_speed=gentle_v,
            spin_angular_speed=spin_w,
            signal_timeout_sec=signal_timeout,
            enable_lidar_safety=enable_lidar,
            obstacle_stop_dist=stop_dist,
            surprise_target_angle_rad=surprise_angle,
            surprise_cooldown_sec=surprise_cooldown,
        )

        # Publishers & Subscribers
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_status = self.create_publisher(String, '/emotion_robot_status', 10)

        self.sub_emotion = self.create_subscription(
            String, '/detected_emotion', self._emotion_callback, 10
        )
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self._odom_callback, 10
        )
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10
        )

        # Configure Create 3 /motion_control safety_override='full' for reverse motion
        self.motion_param_client = self.create_client(
            SetParameters, '/motion_control/set_parameters'
        )
        self.safety_override_configured = False
        self._override_in_flight = False
        self.override_timer = self.create_timer(2.0, self._ensure_safety_override)

        # Control loop at 20 Hz (0.05s default)
        self._last_status_text = ''
        self._last_status_pub_time = 0.0
        self.control_timer = self.create_timer(1.0 / control_rate_hz, self._control_loop_cb)

        self._publish_status('WAITING: Cho tin hieu /detected_emotion...', force=True)
        self.get_logger().info(
            f'emotion_control_node ready (rate={control_rate_hz:.1f}Hz, '
            f'lidar_safety={enable_lidar}, stop_dist={stop_dist:.2f}m, '
            f'watchdog={signal_timeout:.2f}s)'
        )

    def _ensure_safety_override(self) -> None:
        """Ensure Create 3 /motion_control has safety_override='full' so reverse works."""
        if self.safety_override_configured or self._override_in_flight:
            return
        if not self.motion_param_client.service_is_ready():
            return

        req = SetParameters.Request()
        param = Parameter()
        param.name = 'safety_override'
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_STRING,
            string_value=self.safety_override_value,
        )
        req.parameters = [param]
        self._override_in_flight = True
        future = self.motion_param_client.call_async(req)
        future.add_done_callback(self._on_safety_override_done)

    def _on_safety_override_done(self, future) -> None:
        self._override_in_flight = False
        try:
            resp = future.result()
            if resp and all(r.successful for r in resp.results):
                self.safety_override_configured = True
                self.override_timer.cancel()
                self.get_logger().info(
                    f"Configured /motion_control safety_override='{self.safety_override_value}'."
                )
            else:
                self.get_logger().warn(
                    'Failed to set /motion_control safety_override; will retry.'
                )
        except Exception as exc:
            self.get_logger().warn(f'Error setting safety_override: {exc}')

    def _emotion_callback(self, msg: String) -> None:
        self.controller.on_emotion(msg.data, time.monotonic())

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.controller.on_odom(yaw, time.monotonic())

    def _scan_callback(self, msg: LaserScan) -> None:
        front_dist, rear_dist = compute_scan_clearances(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
        )
        self.controller.on_scan(front_dist, rear_dist)

    def publish_stop(self) -> None:
        """Publish a zero-velocity Twist to /cmd_vel safely."""
        try:
            self.pub_cmd_vel.publish(Twist())
        except Exception:
            pass

    def _publish_status(self, text: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and text == self._last_status_text and (now - self._last_status_pub_time) < 0.5:
            return
        self._last_status_text = text
        self._last_status_pub_time = now
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)

    def _control_loop_cb(self) -> None:
        now = time.monotonic()
        lin_x, ang_z, status_text = self.controller.step(now)

        cmd = Twist()
        cmd.linear.x = float(lin_x)
        cmd.angular.z = float(ang_z)
        self.pub_cmd_vel.publish(cmd)

        status_changed = status_text != self._last_status_text
        self._publish_status(status_text, force=status_changed)
        if status_changed:
            if status_text.startswith('STOPPED (Watchdog)') or status_text.startswith('STOPPED (Front') or status_text.startswith('STOPPED (Rear'):
                self.get_logger().warn(status_text)
            else:
                self.get_logger().info(status_text)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EmotionControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
