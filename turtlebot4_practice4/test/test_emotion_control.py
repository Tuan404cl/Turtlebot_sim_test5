# -*- coding: utf-8 -*-
"""Comprehensive pytest suite for emotion_recognition_node and emotion_control_node."""

import math
import time

import pytest
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from turtlebot4_practice4.emotion_control_node import (
    EmotionControlNode,
    EmotionMotionController,
    apply_lidar_safety,
    compute_scan_clearances,
    compute_twist_for_emotion,
    normalize_angle,
    normalize_emotion,
    quaternion_to_yaw,
)
from turtlebot4_practice4.emotion_recognition_node import (
    TARGET_EMOTIONS,
    EmotionStabilizer,
    canonicalize_emotion_scores,
    parse_deepface_result,
)


def _scores(**overrides) -> dict:
    """Build a normalized 6-emotion probability dict for stabilizer tests."""
    base = {label: 0.0 for label in TARGET_EMOTIONS}
    base.update(overrides)
    total = sum(base.values())
    if total <= 0.0:
        base['Neutral'] = 1.0
        return base
    return {k: v / total for k, v in base.items()}


def _make_parsed(emotion: str, confidence: float = 0.85, face_detected: bool = True) -> dict:
    """Create a synthetic parsed DeepFace result matching parse_deepface_result()."""
    if not face_detected:
        return {
            'face_detected': False,
            'bbox': None,
            'raw_emotion': 'Neutral',
            'scores': canonicalize_emotion_scores(None),
            'face_confidence': 0.0,
        }
    rem = max(0.0, 1.0 - confidence)
    scores = {label: (confidence if label == emotion else rem / (len(TARGET_EMOTIONS) - 1)) for label in TARGET_EMOTIONS}
    return {
        'face_detected': True,
        'bbox': (10, 20, 100, 100),
        'raw_emotion': emotion,
        'scores': scores,
        'face_confidence': 0.99,
    }


def _yaw_to_quat(yaw: float):
    """Return (x, y, z, w) quaternion for planar yaw."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _make_scan(front_dist: float = 5.0, rear_dist: float = 5.0, side_dist: float = 5.0) -> LaserScan:
    """Construct a 360-degree LaserScan (-pi to +pi, 1 deg step) with configurable sector clearances."""
    msg = LaserScan()
    msg.angle_min = -math.pi
    msg.angle_increment = math.radians(1.0)
    msg.range_min = 0.05
    msg.range_max = 10.0
    ranges = []
    angle = msg.angle_min
    for _ in range(360):
        ang_deg = abs(math.degrees(normalize_angle(angle)))
        if ang_deg <= 30.0:
            ranges.append(front_dist)
        elif ang_deg >= 150.0:
            ranges.append(rear_dist)
        else:
            ranges.append(side_dist)
        angle += msg.angle_increment
    msg.ranges = ranges
    return msg


# ---------------------------------------------------------------------------
# 1. Normalization & DeepFace Parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'raw,expected',
    [
        ('Happy', 'Happy'),
        ('happy', 'Happy'),
        ('HAPPY', 'Happy'),
        ('  Sad  ', 'Sad'),
        ('SAD', 'Sad'),
        ('neutral', 'Neutral'),
        ('NEUTRAL', 'Neutral'),
        ('angry', 'Angry'),
        ('ANGER', 'Angry'),
        ('surprise', 'Surprise'),
        ('SuRpRiSe', 'Surprise'),
        ('surprised', 'Surprise'),
        ('fear', 'Fear'),
        ('FEARFUL', 'Fear'),
        ('scared', 'Fear'),
        ('disgust', 'Neutral'),
        ('DISGUST', 'Neutral'),
        ('vui ve', 'Happy'),
        ('tuc gian', 'Angry'),
        ('unknown_emotion', 'Neutral'),
        ('confused', 'Neutral'),
        ('', 'Neutral'),
        ('   ', 'Neutral'),
        (None, 'Neutral'),
        (42, 'Neutral'),
    ],
)
def test_normalize_emotion(raw, expected):
    assert normalize_emotion(raw) == expected


def test_canonicalize_emotion_scores_merges_disgust_and_normalizes():
    scores = canonicalize_emotion_scores({
        'happy': 20.0,
        'sad': 10.0,
        'neutral': 30.0,
        'disgust': 20.0,
        'angry': 10.0,
        'surprise': 5.0,
        'fear': 5.0,
        'unknown': 999.0,
    })
    assert set(scores.keys()) == set(TARGET_EMOTIONS)
    assert math.isclose(sum(scores.values()), 1.0, rel_tol=1e-6)
    assert math.isclose(scores['Neutral'], 0.50, rel_tol=1e-6)
    assert math.isclose(scores['Happy'], 0.20, rel_tol=1e-6)

    fallback = canonicalize_emotion_scores(None)
    assert fallback['Neutral'] == 1.0
    assert sum(fallback.values()) == 1.0


def test_parse_deepface_result_no_face_and_disgust():
    no_face = parse_deepface_result(
        [{'region': {'x': 0, 'y': 0, 'w': 640, 'h': 480}, 'face_confidence': 0.0, 'dominant_emotion': 'happy', 'emotion': {'happy': 99.0}}],
        frame_shape=(480, 640),
    )
    assert no_face['face_detected'] is False
    assert no_face['raw_emotion'] == 'Neutral'

    full_frame_no_conf = parse_deepface_result(
        [{'region': {'x': 0, 'y': 0, 'w': 640, 'h': 480}, 'dominant_emotion': 'sad', 'emotion': {'sad': 90.0}}],
        frame_shape=(480, 640),
    )
    assert full_frame_no_conf['face_detected'] is False

    disgust_face = parse_deepface_result(
        [{'region': {'x': 50, 'y': 60, 'w': 120, 'h': 120}, 'face_confidence': 0.92, 'dominant_emotion': 'disgust', 'emotion': {'disgust': 80.0, 'neutral': 20.0}}],
        frame_shape=(480, 640),
    )
    assert disgust_face['face_detected'] is True
    assert disgust_face['raw_emotion'] == 'Neutral'
    assert math.isclose(disgust_face['scores']['Neutral'], 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 2. EmotionStabilizer (smoothing, min_confidence, debounce, face_lost)
# ---------------------------------------------------------------------------

def test_stabilizer_smoothing_and_debounce():
    stab = EmotionStabilizer(
        smoothing_window=4,
        debounce_frames=2,
        min_confidence=0.30,
        face_lost_timeout_sec=1.5,
    )

    # Frame 1: Happy -> debounce_frames=2, so still Neutral after 1 frame
    res1 = stab.update(_make_parsed('Happy', 0.90), now=1.0, has_new_inference=True)
    assert res1['emotion'] == 'Neutral'
    assert res1['face_detected'] is True

    # Frame 2: Happy -> reaches 2 consecutive frames -> switches to Happy
    res2 = stab.update(_make_parsed('Happy', 0.90), now=1.1, has_new_inference=True)
    assert res2['emotion'] == 'Happy'
    assert res2['confidence'] >= 0.80

    # Brief single-frame Sad glitch does not immediately flip away from Happy
    res3 = stab.update(_make_parsed('Sad', 0.60), now=1.2, has_new_inference=True)
    assert res3['emotion'] == 'Happy'


def test_stabilizer_sliding_window_averages_probabilities():
    stab = EmotionStabilizer(
        smoothing_window=4,
        debounce_frames=1,
        min_confidence=0.30,
        face_lost_timeout_sec=1.5,
    )
    stab.update({'face_detected': True, 'raw_emotion': 'Happy', 'scores': _scores(Happy=0.8, Sad=0.2)}, now=0.1)
    res = stab.update({'face_detected': True, 'raw_emotion': 'Sad', 'scores': _scores(Happy=0.2, Sad=0.8)}, now=0.2)
    assert math.isclose(res['scores']['Happy'], 0.5, rel_tol=1e-6)
    assert math.isclose(res['scores']['Sad'], 0.5, rel_tol=1e-6)


def test_stabilizer_min_confidence_gate():
    stab = EmotionStabilizer(
        smoothing_window=2,
        debounce_frames=1,
        min_confidence=0.40,
        face_lost_timeout_sec=1.5,
    )
    # Max emotion is Fear at 0.25 (< 0.40 min_confidence) -> falls back to Neutral
    low_conf = {
        'face_detected': True,
        'raw_emotion': 'Fear',
        'scores': {'Happy': 0.20, 'Sad': 0.15, 'Neutral': 0.10, 'Angry': 0.15, 'Surprise': 0.15, 'Fear': 0.25},
    }
    res = stab.update(low_conf, now=1.0, has_new_inference=True)
    assert res['emotion'] == 'Neutral'


def test_stabilizer_face_lost_timeout():
    stab = EmotionStabilizer(
        smoothing_window=2,
        debounce_frames=1,
        min_confidence=0.30,
        face_lost_timeout_sec=1.5,
    )
    res = stab.update(_make_parsed('Angry', 0.90), now=10.0, has_new_inference=True)
    assert res['emotion'] == 'Angry'

    # No face detected at t=11.0 (1.0s elapsed < 1.5s): hold last stable emotion
    res_wait = stab.update(_make_parsed('Neutral', face_detected=False), now=11.0, has_new_inference=True)
    assert res_wait['emotion'] == 'Angry'
    assert res_wait['face_lost'] is False

    # Poll without new inference at t=11.6 (1.6s elapsed >= 1.5s): reverts to Neutral
    res_lost = stab.update(None, now=11.6, has_new_inference=False)
    assert res_lost['emotion'] == 'Neutral'
    assert res_lost['face_lost'] is True


# ---------------------------------------------------------------------------
# 3. Exact Twist Output for All 6 Emotions
# ---------------------------------------------------------------------------

def test_compute_twist_for_all_six_emotions():
    lin, ang = compute_twist_for_emotion('Happy')
    assert lin > 0.0 and ang > 0.0
    assert (lin, ang) == pytest.approx((0.18, 0.55))

    lin, ang = compute_twist_for_emotion('Sad')
    assert lin > 0.0 and ang < 0.0
    assert (lin, ang) == pytest.approx((0.18, -0.55))

    lin, ang = compute_twist_for_emotion('Neutral')
    assert (lin, ang) == (0.0, 0.0)

    lin, ang = compute_twist_for_emotion('Angry')
    assert lin < 0.0 and ang == 0.0
    assert (lin, ang) == pytest.approx((-0.12, 0.0))

    lin, ang = compute_twist_for_emotion('Surprise', surprise_active=True)
    assert lin == 0.0 and ang != 0.0
    assert (lin, ang) == pytest.approx((0.0, 1.0))

    lin, ang = compute_twist_for_emotion('Surprise', surprise_active=False)
    assert (lin, ang) == (0.0, 0.0)

    lin, ang = compute_twist_for_emotion('Fear')
    assert lin > 0.0 and ang == 0.0
    assert (lin, ang) == pytest.approx((0.12, 0.0))


# ---------------------------------------------------------------------------
# 4. Surprise 360-deg Spin (Odom Wrapping, Fallback Timer, Latch, Neutral Cancel)
# ---------------------------------------------------------------------------

def test_surprise_odom_yaw_integration_across_pi_boundary():
    ctrl = EmotionMotionController(spin_angular_speed=1.0)
    ctrl.on_odom(0.0, now=0.0)
    ctrl.on_emotion('Surprise', now=0.1)

    lin, ang, status = ctrl.step(0.15)
    assert lin == 0.0 and ang == pytest.approx(1.0)
    assert 'EXECUTING [Surprise' in status

    # Step yaw across +pi -> -pi boundary in ~60-degree increments to reach 2*pi
    yaw_steps = [1.0, 2.0, 3.0, -2.2831853, -1.2831853, -0.2831853, 0.01]
    t = 0.2
    for yaw in yaw_steps[:-1]:
        t += 0.2
        ctrl.on_odom(yaw, now=t)
        ctrl.on_emotion('Surprise', now=t)  # keep emotion alive
        lin, ang, _ = ctrl.step(t)
        assert (lin, ang) == pytest.approx((0.0, 1.0))

    # Final step crosses 2*pi accumulated rotation -> stops and latches
    t += 0.2
    ctrl.on_odom(yaw_steps[-1], now=t)
    lin, ang, status = ctrl.step(t)
    assert (lin, ang) == (0.0, 0.0)
    assert 'COMPLETED [Surprise' in status

    # Holding Surprise after completion remains stopped (edge-triggered latch)
    t += 0.2
    ctrl.on_emotion('Surprise', now=t)
    lin, ang, _ = ctrl.step(t)
    assert (lin, ang) == (0.0, 0.0)


def test_surprise_negative_spin_direction_with_odom():
    ctrl = EmotionMotionController(spin_angular_speed=-1.5)
    ctrl.on_odom(0.0, now=0.0)
    ctrl.on_emotion('Surprise', now=0.1)
    lin, ang, _ = ctrl.step(0.15)
    assert (lin, ang) == pytest.approx((0.0, -1.5))

    # Clockwise increments crossing -pi -> +pi
    for i, yaw in enumerate([-1.1, -2.2, 2.9831853, 1.8831853, 0.7831853, -0.05], start=1):
        t = 0.1 + i * 0.2
        ctrl.on_emotion('Surprise', now=t)
        ctrl.on_odom(yaw, now=t)
    lin, ang, _ = ctrl.step(1.5)
    assert (lin, ang) == (0.0, 0.0)


def test_surprise_fallback_timer_without_odom():
    ctrl = EmotionMotionController(spin_angular_speed=2.0, signal_timeout_sec=1.5)  # 2*pi / 2.0 = pi (~3.1416s)
    ctrl.on_emotion('Surprise', now=10.0)

    # Keep /detected_emotion alive (< 1.5s watchdog) while spinning without /odom
    for t in (10.5, 11.0, 11.5, 12.0, 12.5, 13.0):
        ctrl.on_emotion('Surprise', now=t)
        lin, ang, _ = ctrl.step(t)
        assert (lin, ang) == pytest.approx((0.0, 2.0))

    # After >= pi seconds (3.2s >= 3.1416s) -> completes 360 spin via fallback timer
    lin, ang, status = ctrl.step(13.2)
    assert (lin, ang) == (0.0, 0.0)
    assert 'COMPLETED [Surprise' in status


def test_surprise_cancelled_immediately_by_neutral_and_retriggers():
    ctrl = EmotionMotionController(spin_angular_speed=1.0)
    ctrl.on_emotion('Surprise', now=1.0)
    lin, ang, _ = ctrl.step(1.1)
    assert (lin, ang) == pytest.approx((0.0, 1.0))

    # Immediate cancellation by Neutral
    ctrl.on_emotion('Neutral', now=1.2)
    lin, ang, status = ctrl.step(1.25)
    assert (lin, ang) == (0.0, 0.0)
    assert 'Neutral' in status
    assert ctrl.surprise_spinning is False
    assert ctrl.surprise_latched is False

    # Re-triggering Surprise after Neutral starts a new spin
    ctrl.on_emotion('Surprise', now=1.3)
    lin, ang, _ = ctrl.step(1.35)
    assert (lin, ang) == pytest.approx((0.0, 1.0))


# ---------------------------------------------------------------------------
# 5. Watchdog Timeout
# ---------------------------------------------------------------------------

def test_watchdog_stops_robot_and_recovers():
    ctrl = EmotionMotionController(signal_timeout_sec=1.5)
    # Initial state before any message
    lin, ang, status = ctrl.step(0.5)
    assert (lin, ang) == (0.0, 0.0)
    assert 'WAITING' in status

    ctrl.on_emotion('Happy', now=1.0)
    lin, ang, _ = ctrl.step(2.0)  # 1.0s <= 1.5s
    assert (lin, ang) == pytest.approx((0.18, 0.55))

    # Timeout (> 1.5s since t=1.0)
    lin, ang, status = ctrl.step(2.6)
    assert (lin, ang) == (0.0, 0.0)
    assert 'Watchdog' in status
    assert ctrl.watchdog_timed_out is True

    # Also cancels active Surprise spin on timeout
    ctrl.on_emotion('Surprise', now=3.0)
    assert ctrl.step(3.2)[:2] == pytest.approx((0.0, 1.0))
    lin, ang, status = ctrl.step(4.6)
    assert (lin, ang) == (0.0, 0.0)
    assert 'Watchdog' in status
    assert ctrl.surprise_spinning is False


# ---------------------------------------------------------------------------
# 6. LiDAR /scan Safety
# ---------------------------------------------------------------------------

def test_angle_and_quaternion_helpers():
    assert normalize_angle(3.0 * math.pi) == pytest.approx(math.pi)
    assert normalize_angle(-3.0 * math.pi) == pytest.approx(-math.pi)
    for yaw in (-2.5, -0.7, 0.0, 1.2, 2.9):
        qx, qy, qz, qw = _yaw_to_quat(yaw)
        assert quaternion_to_yaw(qx, qy, qz, qw) == pytest.approx(yaw, abs=1e-6)


def test_compute_scan_clearances_filters_invalid_and_separates_sectors():
    scan = _make_scan(front_dist=0.18, rear_dist=0.80, side_dist=0.10)
    # Inject invalid readings in front sector (nan, inf, <= range_min, > range_max)
    mid = len(scan.ranges) // 2  # angle ~ 0 rad (front)
    scan.ranges[mid] = float('nan')
    scan.ranges[mid + 1] = float('inf')
    scan.ranges[mid + 2] = 0.01  # <= range_min (0.05)
    scan.ranges[mid + 3] = 25.0  # > range_max (10.0)

    front, rear = compute_scan_clearances(
        scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max
    )
    assert front == pytest.approx(0.18)
    assert rear == pytest.approx(0.80)


def test_lidar_safety_blocks_front_and_rear_appropriately():
    ctrl = EmotionMotionController(enable_lidar_safety=True, obstacle_stop_dist=0.25)

    # Front obstacle < 0.25m blocks Happy, Sad, Fear; allows Angry (reverse) and Surprise (spin)
    ctrl.on_scan(front_clear_dist=0.20, rear_clear_dist=2.0)
    for fwd_emo in ('Happy', 'Sad', 'Fear'):
        ctrl.on_emotion(fwd_emo, now=1.0)
        lin, ang, status = ctrl.step(1.05)
        assert (lin, ang) == (0.0, 0.0)
        assert 'Front LiDAR' in status

    ctrl.on_emotion('Angry', now=1.1)
    assert ctrl.step(1.15)[:2] == pytest.approx((-0.12, 0.0))

    ctrl.on_emotion('Surprise', now=1.2)
    assert ctrl.step(1.25)[:2] == pytest.approx((0.0, 1.0))

    # Rear obstacle < 0.25m blocks Angry; allows Happy, Sad, Fear
    ctrl.on_scan(front_clear_dist=2.0, rear_clear_dist=0.15)
    ctrl.on_emotion('Angry', now=2.0)
    lin, ang, status = ctrl.step(2.05)
    assert (lin, ang) == (0.0, 0.0)
    assert 'Rear LiDAR' in status

    ctrl.on_emotion('Fear', now=2.1)
    assert ctrl.step(2.15)[:2] == pytest.approx((0.12, 0.0))

    # Pure helper check when disabled
    assert apply_lidar_safety(0.18, 0.55, 0.10, 0.10, enable_lidar_safety=False) == (0.18, 0.55, None)


# ---------------------------------------------------------------------------
# 7. End-to-End ROS 2 Spin Test on EmotionControlNode
# ---------------------------------------------------------------------------

class _HarnessNode(Node):
    """Test helper node publishing /detected_emotion, /odom, /scan and mocking /motion_control/set_parameters."""

    def __init__(self):
        super().__init__('emotion_test_harness')
        self.pub_emotion = self.create_publisher(String, '/detected_emotion', 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', 10)

        self.cmd_vels = []
        self.statuses = []
        self.create_subscription(Twist, '/cmd_vel', lambda m: self.cmd_vels.append(m), 20)
        self.create_subscription(String, '/emotion_robot_status', lambda m: self.statuses.append(m.data), 20)

        self.received_param_calls = []
        self.create_service(SetParameters, '/motion_control/set_parameters', self._handle_set_params)

    def _handle_set_params(self, request, response):
        from rcl_interfaces.msg import SetParametersResult

        for p in request.parameters:
            self.received_param_calls.append((p.name, p.value.type, p.value.string_value))
            res = SetParametersResult()
            res.successful = True
            response.results.append(res)
        return response


def _spin_for(executor: SingleThreadedExecutor, duration_sec: float) -> None:
    end = time.monotonic() + duration_sec
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=0.02)


def test_ros2_end_to_end_emotion_control_node():
    rclpy.init()
    control_node = None
    harness = None
    executor = None
    try:
        control_node = EmotionControlNode()
        harness = _HarnessNode()
        executor = SingleThreadedExecutor()
        executor.add_node(control_node)
        executor.add_node(harness)

        def publish_emotion(label: str, wait_sec: float = 0.18) -> Twist:
            harness.cmd_vels.clear()
            msg = String()
            msg.data = label
            harness.pub_emotion.publish(msg)
            _spin_for(executor, wait_sec)
            assert harness.cmd_vels, f'Expected /cmd_vel messages for {label}'
            return harness.cmd_vels[-1]

        # Ensure clear LiDAR scan first
        harness.pub_scan.publish(_make_scan(front_dist=3.0, rear_dist=3.0))
        _spin_for(executor, 0.1)

        # 1) All 6 emotions -> expected Twist on /cmd_vel
        tw = publish_emotion('Happy')
        assert tw.linear.x == pytest.approx(0.18) and tw.angular.z == pytest.approx(0.55)

        tw = publish_emotion('Sad')
        assert tw.linear.x == pytest.approx(0.18) and tw.angular.z == pytest.approx(-0.55)

        tw = publish_emotion('Angry')
        assert tw.linear.x == pytest.approx(-0.12) and tw.angular.z == pytest.approx(0.0)

        tw = publish_emotion('Fear')
        assert tw.linear.x == pytest.approx(0.12) and tw.angular.z == pytest.approx(0.0)

        tw = publish_emotion('Neutral')
        assert tw.linear.x == pytest.approx(0.0) and tw.angular.z == pytest.approx(0.0)

        # 2) Surprise 360 spin completed via /odom messages across +-pi
        def publish_odom(yaw: float):
            od = Odometry()
            qx, qy, qz, qw = _yaw_to_quat(yaw)
            od.pose.pose.orientation.x = qx
            od.pose.pose.orientation.y = qy
            od.pose.pose.orientation.z = qz
            od.pose.pose.orientation.w = qw
            harness.pub_odom.publish(od)
            _spin_for(executor, 0.06)

        publish_odom(0.0)
        tw = publish_emotion('Surprise')
        assert tw.linear.x == pytest.approx(0.0) and tw.angular.z == pytest.approx(1.0)

        for yaw in (1.1, 2.2, -3.0, -1.9, -0.8, 0.05):
            msg = String()
            msg.data = 'Surprise'
            harness.pub_emotion.publish(msg)
            publish_odom(yaw)

        harness.cmd_vels.clear()
        _spin_for(executor, 0.12)
        assert harness.cmd_vels[-1].linear.x == pytest.approx(0.0)
        assert harness.cmd_vels[-1].angular.z == pytest.approx(0.0)

        # 3) LiDAR front block on Happy
        harness.pub_scan.publish(_make_scan(front_dist=0.15, rear_dist=3.0))
        _spin_for(executor, 0.08)
        tw = publish_emotion('Happy')
        assert tw.linear.x == pytest.approx(0.0) and tw.angular.z == pytest.approx(0.0)

        # LiDAR rear block on Angry
        harness.pub_scan.publish(_make_scan(front_dist=3.0, rear_dist=0.15))
        _spin_for(executor, 0.08)
        tw = publish_emotion('Angry')
        assert tw.linear.x == pytest.approx(0.0) and tw.angular.z == pytest.approx(0.0)

        # Clear scan -> Fear moves forward, then watchdog stops after >1.5s silence
        harness.pub_scan.publish(_make_scan(front_dist=3.0, rear_dist=3.0))
        _spin_for(executor, 0.08)
        tw = publish_emotion('Fear')
        assert tw.linear.x == pytest.approx(0.12)

        _spin_for(executor, 1.65)
        harness.cmd_vels.clear()
        _spin_for(executor, 0.12)
        assert harness.cmd_vels[-1].linear.x == pytest.approx(0.0)
        assert harness.cmd_vels[-1].angular.z == pytest.approx(0.0)

        # 4) Verify Create 3 safety_override='full' service call was issued and accepted
        deadline = time.monotonic() + 2.5
        while not control_node.safety_override_configured and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert control_node.safety_override_configured is True
        assert ('safety_override', ParameterType.PARAMETER_STRING, 'full') in harness.received_param_calls
        assert any('Neutral' in s or 'Watchdog' in s for s in harness.statuses)
    finally:
        if executor is not None:
            executor.shutdown()
        if harness is not None:
            harness.destroy_node()
        if control_node is not None:
            control_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
