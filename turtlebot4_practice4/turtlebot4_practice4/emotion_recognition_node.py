#!/home/tuan/turtelbot4_ws/.venv_emotion/bin/python3
# -*- coding: utf-8 -*-
"""
ROS 2 Node nhận diện cảm xúc khuôn mặt bằng AI (DeepFace + OpenCV)
Node: emotion_recognition_node

Chức năng:
1. Đọc hình ảnh thời gian thực từ Webcam bằng OpenCV (lật gương ngang để dễ quan sát).
2. Chạy suy luận cảm xúc DeepFace (DeepFace.analyze) bất đồng bộ trong background worker thread
   sử dụng buffer 1-slot (latest-frame) để giữ vòng lặp hiển thị OpenCV mượt (~30 FPS).
3. Hỗ trợ 06 cảm xúc chuẩn (title-case): Happy, Sad, Neutral, Angry, Surprise, Fear
   (ánh xạ 'disgust' của DeepFace vào 'Neutral').
4. Áp dụng lọc nhiễu gồm:
   - Làm mượt xác suất theo cửa sổ trượt (smoothing_window)
   - Khử rung theo thời gian (debounce_frames)
   - Watchdog mất khuôn mặt (face_lost_timeout_sec -> tự động về Neutral)
5. Publish kết quả lên các topic:
   - /detected_emotion       (std_msgs/msg/String)
   - /detected_emotion_info  (std_msgs/msg/String - JSON metadata)
   - /emotion/detections/image (sensor_msgs/msg/Image - bgr8 cho RViz2)
6. Subscribe /emotion_robot_status (std_msgs/msg/String) để hiển thị trạng thái Robot lên HUD.
7. Hỗ trợ phím tắt trên cửa sổ OpenCV:
   - q / ESC : Thoát sạch sẽ
   - SPACE   : Bật/tắt dừng khẩn cấp (khóa Neutral)
   - 1 - 6   : Ghi đè cảm xúc tạm thời để kiểm thử (1:Happy, 2:Sad, 3:Neutral, 4:Angry, 5:Surprise, 6:Fear)
   - 0       : Quay lại chế độ AI tự động
"""

import collections
import importlib.util
import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


TARGET_EMOTIONS = ('Happy', 'Sad', 'Neutral', 'Angry', 'Surprise', 'Fear')

CANONICAL_EMOTION_MAP = {
    'happy': 'Happy',
    'sad': 'Sad',
    'neutral': 'Neutral',
    'angry': 'Angry',
    'surprise': 'Surprise',
    'fear': 'Fear',
    'disgust': 'Neutral',
}

# Nhãn tiếng Việt không dấu để hiển thị chuẩn trên OpenCV FONT_HERSHEY_SIMPLEX
EMOTION_VI_LABELS = {
    'Happy': 'Vui ve',
    'Sad': 'Buon',
    'Neutral': 'Binh thuong',
    'Angry': 'Tuc gian',
    'Surprise': 'Ngac nhien',
    'Fear': 'So hai',
}

EMOTION_ACTION_HINTS = {
    'Happy': 'Re vong trai (Curve Left)',
    'Sad': 'Re vong phai (Curve Right)',
    'Neutral': 'Dung tai cho (Stop)',
    'Angry': 'Lui nhe (Move Back)',
    'Surprise': 'Xoay 360 do (Spin 360)',
    'Fear': 'Tien nhe (Move Forward)',
}

# Màu BGR tương ứng cho từng cảm xúc trên HUD
EMOTION_COLORS = {
    'Happy': (0, 220, 90),      # Xanh lá
    'Sad': (235, 160, 40),      # Xanh dương
    'Neutral': (190, 190, 190), # Xám bạc
    'Angry': (40, 50, 245),     # Đỏ
    'Surprise': (0, 220, 255),  # Vàng cam
    'Fear': (235, 100, 210),    # Tím hồng
}

KEY_OVERRIDE_MAP = {
    ord('1'): 'Happy',
    ord('2'): 'Sad',
    ord('3'): 'Neutral',
    ord('4'): 'Angry',
    ord('5'): 'Surprise',
    ord('6'): 'Fear',
}


def canonicalize_emotion_scores(raw_scores: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Chuẩn hóa vector điểm số của DeepFace về đúng 6 nhãn chuẩn (tổng bằng 1.0).
    Cảm xúc 'disgust' được gộp vào 'Neutral'.
    """
    totals = {label: 0.0 for label in TARGET_EMOTIONS}
    if isinstance(raw_scores, dict):
        for key, val in raw_scores.items():
            canonical = CANONICAL_EMOTION_MAP.get(str(key).strip().lower())
            if canonical is None:
                continue
            try:
                num = max(0.0, float(val))
            except (TypeError, ValueError):
                num = 0.0
            totals[canonical] += num

    score_sum = sum(totals.values())
    if score_sum <= 0.0:
        totals['Neutral'] = 1.0
        return totals

    return {label: totals[label] / score_sum for label in TARGET_EMOTIONS}


def parse_deepface_result(
    raw_result: Any,
    frame_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Phân tích kết quả trả về từ DeepFace.analyze thành cấu trúc chuẩn:
    {
        'face_detected': bool,
        'bbox': (x, y, w, h) | None,
        'raw_emotion': str,
        'scores': dict[str, float],
        'face_confidence': float,
    }
    """
    default_scores = canonicalize_emotion_scores(None)
    empty = {
        'face_detected': False,
        'bbox': None,
        'raw_emotion': 'Neutral',
        'scores': default_scores,
        'face_confidence': 0.0,
    }

    if not raw_result:
        return empty

    faces = raw_result if isinstance(raw_result, list) else [raw_result]
    valid_faces = [f for f in faces if isinstance(f, dict)]
    if not valid_faces:
        return empty

    # Chọn khuôn mặt có độ tin cậy cao nhất / diện tích lớn nhất nếu có nhiều người
    def _face_rank(face_dict: Dict[str, Any]) -> Tuple[float, int]:
        conf = face_dict.get('face_confidence')
        conf_val = float(conf) if conf is not None else 0.5
        reg = face_dict.get('region') or {}
        area = max(0, int(reg.get('w', 0))) * max(0, int(reg.get('h', 0)))
        return (conf_val, area)

    best = max(valid_faces, key=_face_rank)
    region = best.get('region') or {}
    try:
        x = int(region.get('x', 0))
        y = int(region.get('y', 0))
        w = int(region.get('w', 0))
        h = int(region.get('h', 0))
    except (TypeError, ValueError):
        return empty

    raw_face_conf = best.get('face_confidence', None)
    face_conf = float(raw_face_conf) if raw_face_conf is not None else None

    if w <= 0 or h <= 0:
        return empty

    # Khi enforce_detection=False và không thấy mặt, DeepFace đặt face_confidence=0
    # hoặc trả về bounding box bằng toàn khung hình.
    if face_conf is not None and face_conf <= 0.0:
        return empty

    if face_conf is None and frame_shape is not None:
        fh, fw = frame_shape[:2]
        if x <= 0 and y <= 0 and w >= fw - 2 and h >= fh - 2:
            return empty

    scores = canonicalize_emotion_scores(best.get('emotion'))
    raw_dom = str(best.get('dominant_emotion', 'neutral')).strip().lower()
    raw_canonical = CANONICAL_EMOTION_MAP.get(raw_dom, 'Neutral')

    return {
        'face_detected': True,
        'bbox': (x, y, w, h),
        'raw_emotion': raw_canonical,
        'scores': scores,
        'face_confidence': face_conf if face_conf is not None else 1.0,
    }


class EmotionStabilizer:
    """
    Bộ ổn định cảm xúc kết hợp:
    - Làm mượt xác suất trên cửa sổ trượt (smoothing_window)
    - Ngưỡng tin cậy tối thiểu (min_confidence)
    - Khử rung qua nhiều khung hình suy luận (debounce_frames)
    - Watchdog mất khuôn mặt (face_lost_timeout_sec -> về Neutral)
    """

    def __init__(
        self,
        smoothing_window: int = 4,
        debounce_frames: int = 2,
        min_confidence: float = 0.30,
        face_lost_timeout_sec: float = 1.5,
    ):
        self.smoothing_window = max(1, int(smoothing_window))
        self.debounce_frames = max(1, int(debounce_frames))
        self.min_confidence = float(min_confidence)
        self.face_lost_timeout_sec = max(0.1, float(face_lost_timeout_sec))

        self.score_history = collections.deque(maxlen=self.smoothing_window)
        self.stable_emotion: str = 'Neutral'
        self.stable_confidence: float = 1.0
        self.raw_emotion: str = 'Neutral'
        self.smoothed_scores: Dict[str, float] = canonicalize_emotion_scores(None)
        self.pending_emotion: Optional[str] = None
        self.pending_count: int = 0
        self.last_face_time: Optional[float] = None
        self.face_detected: bool = False

    def update(
        self,
        parsed: Optional[Dict[str, Any]],
        now: float,
        has_new_inference: bool = True,
    ) -> Dict[str, Any]:
        """
        Cập nhật trạng thái ổn định.
        - Chỉ thêm mẫu mới vào bộ đệm làm mượt / debounce khi has_new_inference=True.
        - Luôn kiểm tra watchdog mất khuôn mặt theo thời gian thực `now`.
        """
        if has_new_inference and parsed is not None:
            self.face_detected = bool(parsed.get('face_detected', False))
            if self.face_detected:
                self.last_face_time = now
                scores = parsed.get('scores') or canonicalize_emotion_scores(None)
                self.raw_emotion = str(parsed.get('raw_emotion', 'Neutral'))
                self.score_history.append(scores)

                # Trung bình cộng xác suất trên cửa sổ trượt
                avg_scores = {}
                n = float(len(self.score_history))
                for label in TARGET_EMOTIONS:
                    avg_scores[label] = sum(s.get(label, 0.0) for s in self.score_history) / n
                self.smoothed_scores = avg_scores

                top_label = max(TARGET_EMOTIONS, key=lambda lbl: avg_scores[lbl])
                top_conf = avg_scores[top_label]
                candidate = top_label if top_conf >= self.min_confidence else 'Neutral'
                candidate_conf = top_conf if candidate != 'Neutral' else avg_scores['Neutral']

                if candidate == self.pending_emotion:
                    self.pending_count += 1
                else:
                    self.pending_emotion = candidate
                    self.pending_count = 1

                if self.pending_count >= self.debounce_frames:
                    self.stable_emotion = candidate
                    self.stable_confidence = candidate_conf
                elif self.stable_emotion == candidate:
                    self.stable_confidence = candidate_conf

        # Kiểm tra Watchdog mất khuôn mặt
        face_lost = (
            self.last_face_time is None
            or (now - self.last_face_time) >= self.face_lost_timeout_sec
        )
        if not self.face_detected and face_lost:
            self.score_history.clear()
            self.pending_emotion = None
            self.pending_count = 0
            self.stable_emotion = 'Neutral'
            self.stable_confidence = 1.0
            self.raw_emotion = 'Neutral'
            self.smoothed_scores = canonicalize_emotion_scores(None)

        return {
            'emotion': self.stable_emotion,
            'raw_emotion': self.raw_emotion,
            'confidence': float(self.stable_confidence),
            'scores': dict(self.smoothed_scores),
            'face_detected': self.face_detected,
            'face_lost': face_lost and not self.face_detected,
        }


class EmotionRecognitionNode(Node):
    """
    ROS 2 Node: emotion_recognition_node
    Nhận diện 6 cảm xúc khuôn mặt bằng DeepFace bất đồng bộ và xuất bản qua ROS 2 topics.
    """

    def __init__(self):
        super().__init__('emotion_recognition_node')

        # Khai báo các tham số ROS 2
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('detector_backend', 'opencv')
        self.declare_parameter('min_confidence', 0.30)
        self.declare_parameter('smoothing_window', 4)
        self.declare_parameter('debounce_frames', 2)
        self.declare_parameter('face_lost_timeout_sec', 1.5)
        self.declare_parameter('publish_rate_hz', 15.0)
        self.declare_parameter('headless', False)
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)

        self.camera_id = int(self.get_parameter('camera_id').value)
        self.frame_width = int(self.get_parameter('frame_width').value)
        self.frame_height = int(self.get_parameter('frame_height').value)
        self.detector_backend = str(self.get_parameter('detector_backend').value).strip() or 'opencv'
        self.min_confidence = float(self.get_parameter('min_confidence').value)
        self.smoothing_window = int(self.get_parameter('smoothing_window').value)
        self.debounce_frames = int(self.get_parameter('debounce_frames').value)
        self.face_lost_timeout_sec = float(self.get_parameter('face_lost_timeout_sec').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.headless = bool(self.get_parameter('headless').value)
        self.use_sim_time = bool(self.get_parameter('use_sim_time').value)

        # Publishers
        self.pub_emotion = self.create_publisher(String, '/detected_emotion', 10)
        self.pub_emotion_info = self.create_publisher(String, '/detected_emotion_info', 10)
        self.pub_emotion_img = self.create_publisher(Image, '/emotion/detections/image', 5)

        # Subscriber trạng thái robot
        self.robot_status_text = 'IDLE: Dang cho tin hieu cam xuc...'
        self.sub_robot_status = self.create_subscription(
            String,
            '/emotion_robot_status',
            self._robot_status_cb,
            10,
        )

        # Bộ ổn định cảm xúc
        self.stabilizer = EmotionStabilizer(
            smoothing_window=self.smoothing_window,
            debounce_frames=self.debounce_frames,
            min_confidence=self.min_confidence,
            face_lost_timeout_sec=self.face_lost_timeout_sec,
        )

        # Trạng thái điều khiển / ghi đè bàn phím
        self.emergency_stop = False
        self.manual_override: Optional[str] = None

        # Mở Webcam bằng OpenCV (xử lý an toàn nếu không có camera)
        self.cap = cv2.VideoCapture(self.camera_id)
        self.camera_ok = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera_ok = True
            self.get_logger().info(
                f'Đã kết nối Webcam (/dev/video{self.camera_id}) ở {self.frame_width}x{self.frame_height}.'
            )
        else:
            self.get_logger().warn(
                f'Không thể mở /dev/video{self.camera_id}! '
                'Chạy ở chế độ khung hình giả lập (vẫn hỗ trợ phím tắt 1-6 và SPACE).'
            )

        # Đồng bộ hóa cho worker thread DeepFace (1-slot latest-frame buffer)
        self._stop_event = threading.Event()
        self._frame_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._pending_frame: Optional[np.ndarray] = None

        self._result_lock = threading.Lock()
        self._latest_inference: Optional[Dict[str, Any]] = None
        self._inference_seq: int = 0
        self._last_consumed_seq: int = 0
        self._model_ready: bool = False
        self._model_error: Optional[str] = None
        self._inference_fps: float = 0.0

        self._worker_thread = threading.Thread(
            target=self._inference_worker_loop,
            name='deepface_emotion_worker',
            daemon=True,
        )
        self._worker_thread.start()

        # Trạng thái hiển thị và publish
        self.prev_frame_time = time.time()
        self.fps = 0.0
        self.last_publish_time = 0.0
        self.last_logged_emotion: Optional[str] = None
        self._shutdown_requested = False

        # Timer ~30 FPS đọc camera & cập nhật HUD
        self.timer = self.create_timer(1.0 / 30.0, self._process_frame_cb)

        self.get_logger().info('====================================================================')
        self.get_logger().info(' NODE NHẬN DIỆN CẢM XÚC AI (emotion_recognition_node) ĐÃ KHỞI CHẠY! ')
        self.get_logger().info(f' Detector backend: {self.detector_backend} | Min conf: {self.min_confidence:.2f}')
        self.get_logger().info(' Topics publish : /detected_emotion, /detected_emotion_info, /emotion/detections/image')
        self.get_logger().info(' Phím tắt HUD   : 1-6 (Test cảm xúc) | 0 (AI) | SPACE (Dừng E-Stop) | Q/ESC (Thoát)')
        self.get_logger().info('====================================================================')

    def _robot_status_cb(self, msg: String):
        self.robot_status_text = msg.data

    def _submit_frame_for_inference(self, bgr_frame: np.ndarray):
        """Ghi đè khung hình mới nhất vào buffer 1-slot cho worker thread."""
        with self._frame_lock:
            self._pending_frame = bgr_frame.copy()
        self._frame_event.set()

    def _inference_worker_loop(self):
        """Luồng nền tải DeepFace và chạy suy luận cảm xúc không chặn luồng OpenCV."""
        try:
            from deepface import DeepFace  # noqa: WPS433

            # Warm-up model trên ảnh nhỏ để suy luận đầu tiên không bị trễ lớn
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            DeepFace.analyze(
                dummy,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                silent=True,
            )
            self._model_ready = True
            self.get_logger().info('DeepFace emotion model đã sẵn sàng.')
        except Exception as exc:
            self._model_error = str(exc)
            self.get_logger().error(f'Không thể tải DeepFace model: {exc}')
            return

        while not self._stop_event.is_set():
            triggered = self._frame_event.wait(timeout=0.2)
            if self._stop_event.is_set():
                break
            if not triggered:
                continue

            with self._frame_lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._frame_event.clear()

            if frame is None:
                continue

            t0 = time.time()
            try:
                raw_result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend=self.detector_backend,
                    silent=True,
                )
                parsed = parse_deepface_result(raw_result, frame.shape[:2])
            except Exception as exc:
                self.get_logger().debug(f'Lỗi suy luận DeepFace: {exc}')
                parsed = parse_deepface_result(None, frame.shape[:2])

            elapsed = max(time.time() - t0, 1e-4)
            with self._result_lock:
                self._latest_inference = parsed
                self._inference_seq += 1
                self._inference_fps = 0.7 * self._inference_fps + 0.3 * (1.0 / elapsed)

    def _handle_keyboard(self, key: int):
        """Xử lý phím tắt trên cửa sổ OpenCV."""
        if key in (ord('q'), 27):
            self.get_logger().info('Đóng node nhận diện cảm xúc theo yêu cầu người dùng (Q/ESC).')
            self._shutdown_requested = True
            if rclpy.ok():
                rclpy.shutdown()
        elif key == 32:  # SPACE
            self.emergency_stop = not self.emergency_stop
            state_str = 'BAT (Khoa Neutral)' if self.emergency_stop else 'TAT'
            self.get_logger().warn(f'[E-STOP] Dừng khẩn cấp: {state_str}')
        elif key == ord('0'):
            if self.manual_override is not None:
                self.get_logger().info('[CHE DO] Quay lai che do nhan dien AI tu dong.')
            self.manual_override = None
        elif key in KEY_OVERRIDE_MAP:
            chosen = KEY_OVERRIDE_MAP[key]
            self.manual_override = chosen
            self.emergency_stop = False
            self.get_logger().info(f'[OVERRIDE] Tam thoi chon cam xuc: {chosen}')

    def _draw_hud(
        self,
        frame: np.ndarray,
        active_emotion: str,
        confidence: float,
        scores: Dict[str, float],
        bbox: Optional[Tuple[int, int, int, int]],
        face_detected: bool,
        mode_label: str,
    ):
        """Vẽ Bounding Box khuôn mặt, nhãn EN/VI, biểu đồ 6 thanh xác suất và trạng thái Robot."""
        h, w = frame.shape[:2]
        color = EMOTION_COLORS.get(active_emotion, (200, 200, 200))
        vi_label = EMOTION_VI_LABELS.get(active_emotion, '')

        # 1. Vẽ Bounding Box khuôn mặt (khi ở chế độ AI và phát hiện khuôn mặt)
        if face_detected and bbox is not None and not self.emergency_stop and self.manual_override is None:
            bx, by, bw, bh = bbox
            x1 = max(0, min(w - 1, bx))
            y1 = max(0, min(h - 1, by))
            x2 = max(x1 + 1, min(w - 1, bx + bw))
            y2 = max(y1 + 1, min(h - 1, by + bh))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            box_label = f'{active_emotion} ({vi_label}) {confidence * 100.0:.1f}%'
            (tw, th), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            lbl_y = max(y1 - 8, th + 12)
            cv2.rectangle(frame, (x1, lbl_y - th - 8), (x1 + tw + 12, lbl_y + 4), color, -1)
            cv2.putText(
                frame,
                box_label,
                (x1 + 6, lbl_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (15, 15, 15),
                2,
                cv2.LINE_AA,
            )

        # 2. Panel thông tin phía trên (Header HUD)
        header_overlay = frame.copy()
        cv2.rectangle(header_overlay, (8, 8), (w - 8, 126), (18, 18, 24), -1)
        cv2.addWeighted(header_overlay, 0.72, frame, 0.28, 0, frame)

        if self._model_error:
            ai_state = 'AI: ERROR'
        elif not self._model_ready:
            ai_state = 'AI: Loading DeepFace...'
        else:
            ai_state = f'AI: {self._inference_fps:.1f}Hz ({self.detector_backend})'

        cv2.putText(
            frame,
            f'DEEPFACE EMOTION | FPS: {self.fps:.1f} | {ai_state} | [{mode_label}]',
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255) if not self.emergency_stop else (0, 80, 255),
            2 if self.emergency_stop else 1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f'Emotion: {active_emotion} ({vi_label})  |  Conf: {confidence * 100.0:.1f}%',
            (16, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            color,
            2,
            cv2.LINE_AA,
        )

        hint = EMOTION_ACTION_HINTS.get(active_emotion, '')
        cv2.putText(
            frame,
            f'Command: {hint}',
            (16, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f'Robot  : {self.robot_status_text[:68]}',
            (16, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 255, 140),
            1,
            cv2.LINE_AA,
        )

        # 3. Biểu đồ thanh xác suất 6 cảm xúc (bên phải)
        panel_w = 215
        panel_h = 158
        px1 = max(8, w - panel_w - 8)
        py1 = 134
        px2 = w - 8
        py2 = min(h - 46, py1 + panel_h)
        if py2 > py1 + 60:
            bar_overlay = frame.copy()
            cv2.rectangle(bar_overlay, (px1, py1), (px2, py2), (18, 18, 24), -1)
            cv2.addWeighted(bar_overlay, 0.70, frame, 0.30, 0, frame)

            cv2.putText(
                frame,
                'Probabilities (6)',
                (px1 + 8, py1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (210, 225, 245),
                1,
                cv2.LINE_AA,
            )

            row_y = py1 + 36
            bar_max_w = 90
            for label in TARGET_EMOTIONS:
                prob = float(scores.get(label, 0.0))
                bar_color = EMOTION_COLORS[label] if label == active_emotion else (120, 120, 130)
                text_color = EMOTION_COLORS[label] if label == active_emotion else (210, 210, 210)
                cv2.putText(
                    frame,
                    f'{label:<8}',
                    (px1 + 8, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.39,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
                bx_start = px1 + 76
                cv2.rectangle(
                    frame,
                    (bx_start, row_y - 9),
                    (bx_start + bar_max_w, row_y + 1),
                    (65, 65, 75),
                    1,
                )
                fill_w = int(bar_max_w * min(1.0, max(0.0, prob)))
                if fill_w > 0:
                    cv2.rectangle(
                        frame,
                        (bx_start, row_y - 9),
                        (bx_start + fill_w, row_y + 1),
                        bar_color,
                        -1,
                    )
                cv2.putText(
                    frame,
                    f'{prob * 100.0:4.0f}%',
                    (bx_start + bar_max_w + 5, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.37,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
                row_y += 20

        # 4. Thanh hướng dẫn phím tắt phía dưới
        footer = frame.copy()
        cv2.rectangle(footer, (8, h - 38), (w - 8, h - 8), (18, 18, 24), -1)
        cv2.addWeighted(footer, 0.72, frame, 0.28, 0, frame)
        cv2.putText(
            frame,
            'Keys: [1]Happy [2]Sad [3]Neutral [4]Angry [5]Surprise [6]Fear [0]AI | [SPACE]E-Stop | [Q/ESC]Quit',
            (14, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (215, 240, 255),
            1,
            cv2.LINE_AA,
        )

    def _publish_annotated_image(self, bgr_frame: np.ndarray):
        """Xuất bản khung hình đã chú thích lên /emotion/detections/image (bgr8)."""
        if not rclpy.ok():
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'webcam_optical_frame'
        h, w = bgr_frame.shape[:2]
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(w * 3)
        msg.data = np.ascontiguousarray(bgr_frame).tobytes()
        self.pub_emotion_img.publish(msg)

    def _process_frame_cb(self):
        if self._shutdown_requested or not rclpy.ok():
            return

        now = time.time()
        dt = max(now - self.prev_frame_time, 1e-4)
        self.fps = 0.85 * self.fps + 0.15 * (1.0 / dt)
        self.prev_frame_time = now

        frame_from_cam = False
        if self.cap is not None and self.cap.isOpened():
            ret, bgr_frame = self.cap.read()
            if ret and bgr_frame is not None:
                bgr_frame = cv2.flip(bgr_frame, 1)
                frame_from_cam = True
            else:
                bgr_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        else:
            bgr_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        if frame_from_cam:
            self._submit_frame_for_inference(bgr_frame)
        else:
            cv2.putText(
                bgr_frame,
                f'NO WEBCAM (/dev/video{self.camera_id}) - Use keys 1-6 or SPACE to test',
                (30, self.frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                1,
                cv2.LINE_AA,
            )

        # Lấy kết quả suy luận mới nhất từ worker thread
        with self._result_lock:
            seq = self._inference_seq
            parsed = self._latest_inference

        has_new = seq != self._last_consumed_seq
        if has_new:
            self._last_consumed_seq = seq

        stabilized = self.stabilizer.update(parsed, now, has_new_inference=has_new)

        # Áp dụng chế độ E-STOP hoặc Manual Override nếu đang bật
        if self.emergency_stop:
            active_emotion = 'Neutral'
            raw_emotion = 'E-STOP'
            confidence = 1.0
            scores = canonicalize_emotion_scores({'neutral': 1.0})
            mode_label = 'E-STOP'
        elif self.manual_override is not None:
            active_emotion = self.manual_override
            raw_emotion = f'MANUAL:{self.manual_override}'
            confidence = 1.0
            scores = canonicalize_emotion_scores({self.manual_override.lower(): 1.0})
            mode_label = f'OVERRIDE:{self.manual_override}'
        else:
            active_emotion = stabilized['emotion']
            raw_emotion = stabilized['raw_emotion']
            confidence = stabilized['confidence']
            scores = stabilized['scores']
            mode_label = 'AI'

        # Publish topic /detected_emotion & /detected_emotion_info
        publish_interval = 1.0 / max(1.0, self.publish_rate_hz)
        emotion_changed = active_emotion != self.last_logged_emotion
        if rclpy.ok() and (emotion_changed or (now - self.last_publish_time) >= publish_interval):
            msg = String()
            msg.data = active_emotion
            self.pub_emotion.publish(msg)

            info_msg = String()
            info_msg.data = json.dumps({
                'emotion': active_emotion,
                'raw_emotion': raw_emotion,
                'confidence': round(float(confidence), 4),
                'scores': {k: round(float(v), 4) for k, v in scores.items()},
                'timestamp': round(now, 3),
                'face_detected': bool(stabilized['face_detected']),
                'mode': mode_label,
            })
            self.pub_emotion_info.publish(info_msg)
            self.last_publish_time = now

            if emotion_changed:
                self.last_logged_emotion = active_emotion
                self.get_logger().info(
                    f'[CẢM XÚC] {active_emotion:<8} ({EMOTION_VI_LABELS.get(active_emotion, "")}) | '
                    f'Conf: {confidence * 100.0:5.1f}% | Mode: {mode_label}'
                )

        bbox = parsed.get('bbox') if parsed else None
        face_detected = bool(stabilized['face_detected'])
        self._draw_hud(
            bgr_frame,
            active_emotion,
            confidence,
            scores,
            bbox,
            face_detected,
            mode_label,
        )
        self._publish_annotated_image(bgr_frame)

        if not self.headless and os.environ.get('DISPLAY'):
            cv2.imshow('TurtleBot 4 - AI Emotion Recognition (DeepFace)', bgr_frame)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self._handle_keyboard(key)

    def destroy_node(self):
        self._stop_event.set()
        self._frame_event.set()
        try:
            if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.5)
        except Exception:
            pass
        try:
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()
            if not self.headless and os.environ.get('DISPLAY'):
                cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def _ensure_emotion_venv():
    """Re-exec with .venv_emotion Python if deepface is unavailable in the current interpreter."""
    if importlib.util.find_spec('deepface') is not None:
        return
    if os.environ.get('_EMOTION_NODE_REEXEC') == '1':
        return
    venv_root = '/home/tuan/turtelbot4_ws/.venv_emotion'
    venv_python = os.path.join(venv_root, 'bin', 'python3')
    if (
        os.path.isfile(venv_python)
        and os.access(venv_python, os.X_OK)
        and os.path.abspath(sys.executable) != os.path.abspath(venv_python)
        and os.path.abspath(sys.prefix) != os.path.abspath(venv_root)
    ):
        os.environ['_EMOTION_NODE_REEXEC'] = '1'
        os.execv(venv_python, [venv_python] + sys.argv)


def main(args=None):
    _ensure_emotion_venv()
    rclpy.init(args=args)
    node = EmotionRecognitionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if rclpy.ok():
            node.get_logger().error(f'Lỗi ngoài dự kiến trong emotion_recognition_node: {exc}')
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
