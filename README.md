# Bài Tập Thực Hành – Điều Khiển Robot Tự Hành TurtleBot 4
## Định Vị Bản Đồ (AMCL Localization), Điều Hướng Tự Động (Nav2) & Nhận Diện Vật Thể Với Camera OAK-D + YOLOv8

Dự án thực hành hoàn chỉnh cho robot **TurtleBot 4 Standard** trong môi trường mô phỏng **Gazebo Fortress** trên nền tảng **ROS 2 Humble (Ubuntu 22.04 LTS)**, kết hợp định vị **AMCL**, điều hướng tự động qua 03 tọa độ mục tiêu bằng **Nav2 (`nav2_simple_commander`)** và nhận diện vật thể thời gian thực từ **Camera RGB OAK-D** bằng mô hình **YOLOv8 (`yolov8n.pt`)**.

---

## 📌 Thông Tin Nộp Bài
* **Họ và tên sinh viên:** Nguyễn Quang Tuân
* **Mã số sinh viên (MSSV):** 23020767
* **Giảng viên hướng dẫn:** PGS.TS. Hoàng Văn Xiêm, CN. Nguyễn Quốc Bảo
* **Video Demo (Google Drive - Chế độ công khai):** [Xem Video Demo trên Google Drive](https://drive.google.com/drive/folders/1vl4O0C9A4C3Ed_HwHmJAeVCzwEQZulCx?usp=drive_link)
* **GitHub Repository (Public - Bài 8):** `https://github.com/Tuan404cl/Turtlebot_sim_test5`
* **File báo cáo kỹ thuật Bài tập 8 (Nhận diện Biểu cảm Khuôn mặt & Điều khiển TurtleBot 4 - PDF):** [turtlebot4_practice4/docs/bao_cao_bai_tap_08.pdf](turtlebot4_practice4/docs/bao_cao_bai_tap_08.pdf)
* **File báo cáo kỹ thuật Bài tập 8 (Nhận diện Biểu cảm Khuôn mặt & Điều khiển TurtleBot 4 - Markdown):** [turtlebot4_practice4/docs/bao_cao_bai_tap_08.md](turtlebot4_practice4/docs/bao_cao_bai_tap_08.md)
* **File báo cáo kỹ thuật Bài tập 8 (Nhận diện Biểu cảm Khuôn mặt & Điều khiển TurtleBot 4 - HTML):** [turtlebot4_practice4/docs/bao_cao_bai_tap_08.html](turtlebot4_practice4/docs/bao_cao_bai_tap_08.html)
* **File báo cáo kỹ thuật Bài tập 7 (Phân vùng Bản đồ & Điều hướng Giọng nói - Markdown chuẩn):** [turtlebot4_practice4/docs/bao_cao_bai_tap_07.md](turtlebot4_practice4/docs/bao_cao_bai_tap_07.md)
* **File báo cáo kỹ thuật (Bài tập Phân vùng Bản đồ & Điều hướng Giọng nói - PDF 2 trang):** [turtlebot4_practice4/docs/bao_cao_voice_navigation.pdf](turtlebot4_practice4/docs/bao_cao_voice_navigation.pdf)
* **File báo cáo kỹ thuật (Bài tập Phân vùng Bản đồ & Điều hướng Giọng nói - Markdown):** [turtlebot4_practice4/docs/bao_cao_voice_navigation.md](turtlebot4_practice4/docs/bao_cao_voice_navigation.md)
* **File báo cáo kỹ thuật Bài tập 6 (Điều khiển cử chỉ tay - Markdown):** [turtlebot4_practice4/docs/bao_cao_bai_tap_06.md](turtlebot4_practice4/docs/bao_cao_bai_tap_06.md)
* **Báo cáo kỹ thuật chi tiết (Markdown):** [turtlebot4_practice4/docs/bao_cao_thuc_hanh_04.md](turtlebot4_practice4/docs/bao_cao_thuc_hanh_04.md)
* **Báo cáo kỹ thuật (HTML):** [turtlebot4_practice4/docs/bao_cao_thuc_hanh_04.html](turtlebot4_practice4/docs/bao_cao_thuc_hanh_04.html)
* **Báo cáo kỹ thuật (PDF):** [turtlebot4_practice4/docs/bao_cao_thuc_hanh_04.pdf](turtlebot4_practice4/docs/bao_cao_thuc_hanh_04.pdf)

---

## 📂 Cấu Trúc Gói Chương Trình Trong `src/`

```
turtlebot4_ws/src/
├── turtlebot4_practice4/
│   ├── package.xml                       # Khai báo phụ thuộc package ROS 2
│   ├── setup.py                          # Cấu hình cài đặt package & 07 console_scripts
│   ├── turtlebot4_practice4/
│   │   ├── spawn_yolo_objects.py         # (Phần 1) Tự động thêm mô hình người, ghế, bàn vào Gazebo
│   │   ├── auto_undock.py                # (Phần 1) Tự động điều khiển robot rời trạm sạc (Undock)
│   │   ├── tf_relay.py                   # (Phần 1) Đồng bộ thời gian TF & mock hardware services
│   │   ├── goal_survey.py                # (Phần 2) Khảo sát /clicked_point & /amcl_pose lưu JSON
│   │   ├── auto_nav_goals.py             # (Phần 3) Điều hướng tự động 03 Goals bằng Nav2 Code
│   │   ├── yolo_detector.py              # (Phần 4) Nhận diện vật thể Camera OAK-D + YOLOv8
│   │   └── check_topics.py               # Kiểm tra tự động /scan, /odom, /cmd_vel
│   ├── launch/
│   │   ├── sim_robot.launch.py           # Khởi chạy Gazebo Simulation (TurtleBot 4 Standard)
│   │   ├── sim_localization.launch.py    # Khởi chạy AMCL Localization & Map Server
│   │   └── sim_navigation.launch.py      # Khởi chạy tích hợp: Gazebo + Objects + AMCL + Nav2 + YOLO
│   ├── config/
│   │   ├── surveyed_goals.json           # Tọa độ (x, y, yaw) của 03 Goals khảo sát ở Phần 2
│   │   ├── localization_params.yaml      # Cấu hình tham số AMCL và Map Server
│   │   ├── nav2_params.yaml              # Cấu hình Costmaps, DWB Controller, NavFn Planner
│   │   ├── fastdds_udp_only.xml          # Cấu hình FastDDS UDP-only chống lỗi SHM deadlock
│   │   └── turtlebot4_nav2.rviz          # Giao diện RViz2 cho Localization, Nav2 & Camera YOLO
│   ├── maps/
│   │   └── map.yaml & map.pgm            # Bản đồ 2D đã xây dựng từ tuần trước
│   ├── scripts/
│   │   ├── run_part1_localization.sh     # Chạy Phần 1: Simulation + Load Map + AMCL + Nav2
│   │   ├── run_part2_survey_goals.sh     # Chạy Phần 2: Khảo sát tọa độ 03 Goals trên RViz2
│   │   ├── run_part3_auto_nav.sh         # Chạy Phần 3: Lập trình tự hành qua 03 Goals
│   │   ├── run_part4_yolo.sh             # Chạy Phần 4: Nhận diện vật thể Camera với YOLOv8
│   │   └── run_part4_navigation.sh       # Chạy tích hợp toàn bộ Phần 1 -> Phần 4 trong 1 lệnh
│   └── docs/
│       └── bao_cao_thuc_hanh_04.(md|html|pdf) & report.tex
└── irobot_create_control/                # Package điều khiển động học vi sai cho đế Create 3
```

---

## 🚀 Hướng Dẫn Thực Thi Nhanh

```bash
# 1. Build workspace
cd /home/tuan/turtelbot4_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# 2. Chạy tích hợp Bài tập 8 – Nhận diện Biểu cảm Khuôn mặt (DeepFace + OpenCV) & Điều khiển TurtleBot 4:
./src/turtlebot4_practice4/scripts/run_emotion_control.sh

# 3. Chạy tích hợp Bài tập Phân vùng Bản đồ & Điều hướng Bằng Giọng nói với Nav2:
./src/turtlebot4_practice4/scripts/run_voice_navigation_all.sh

# 4. Chạy tích hợp Bài tập Điều khiển Hành vi TurtleBot 4 bằng Cử chỉ tay AI (MediaPipe + OpenCV):
./src/turtlebot4_practice4/scripts/run_gesture_all.sh

# 5. Hoặc chạy tích hợp Bài tập AMCL Localization + Auto Nav2 + YOLOv8:
./src/turtlebot4_practice4/scripts/run_part4_navigation.sh
```

---

## 😊 Bài Tập Thực Hành Số 8 – Nhận Diện Biểu Cảm Khuôn Mặt (DeepFace + OpenCV) & Điều Khiển Quỹ Đạo TurtleBot 4 Qua `/cmd_vel`

* **Báo cáo kỹ thuật Bài thực hành số 8 (PDF):** [`turtlebot4_practice4/docs/bao_cao_bai_tap_08.pdf`](turtlebot4_practice4/docs/bao_cao_bai_tap_08.pdf)
* **Báo cáo kỹ thuật Bài thực hành số 8 (Markdown):** [`turtlebot4_practice4/docs/bao_cao_bai_tap_08.md`](turtlebot4_practice4/docs/bao_cao_bai_tap_08.md)
* **Báo cáo kỹ thuật Bài thực hành số 8 (HTML):** [`turtlebot4_practice4/docs/bao_cao_bai_tap_08.html`](turtlebot4_practice4/docs/bao_cao_bai_tap_08.html)
* **Ảnh kết quả nhận diện 06 biểu cảm khuôn mặt thực tế:** [`turtlebot4_practice4/docs/figures/emotion_6_faces_grid.png`](turtlebot4_practice4/docs/figures/emotion_6_faces_grid.png)
* **Biểu đồ quỹ đạo Odometry của 06 biểu cảm:** [`turtlebot4_practice4/docs/figures/emotion_trajectories_odom.png`](turtlebot4_practice4/docs/figures/emotion_trajectories_odom.png)
* **Ảnh tổng hợp trình diễn thực tế Gazebo + RViz2 + DeepFace HUD (6 biểu cảm):** [`turtlebot4_practice4/docs/figures/emotion_live_demo_composite.png`](turtlebot4_practice4/docs/figures/emotion_live_demo_composite.png)
* **Biểu đồ telemetry ROS 2 thực tế trong Gazebo (`/odom`, `/cmd_vel`, `Surprise 360°`):** [`turtlebot4_practice4/docs/figures/emotion_live_odom_telemetry.png`](turtlebot4_practice4/docs/figures/emotion_live_odom_telemetry.png)
* **Video Demo thực tế (Local & Google Drive):** `/home/tuan/Videos/emotion_control_demo.mp4` (`2560x1600`, H.264 `yuv420p`, `15 fps`, `1389` khung hình, thời lượng `92.60 s`, dung lượng `32.06 MiB` / `33,621,221 bytes`) – [Xem trên Google Drive](https://drive.google.com/drive/folders/1vl4O0C9A4C3Ed_HwHmJAeVCzwEQZulCx?usp=drive_link)

### Bảng Ánh Xạ 06 Biểu Cảm Khuôn Mặt Sang Lệnh Vận Tốc `/cmd_vel`
| Biểu cảm (Emotion) | Độ tin cậy thực nghiệm | Hành động của TurtleBot 4 | $v_x$ (`m/s`) | $\omega_z$ (`rad/s`) |
| :--- | :---: | :--- | :---: | :---: |
| **`Happy` (Vui vẻ)** | `99.7%` | Di chuyển theo quỹ đạo cong sang trái ($R = 0.327\text{ m}$) | `+0.18` | `+0.55` |
| **`Sad` (Buồn)** | `59.8%` | Di chuyển theo quỹ đạo cong sang phải ($R = 0.327\text{ m}$) | `+0.18` | `-0.55` |
| **`Neutral` (Bình thường)** | `40.8%` | Dừng tại chỗ (Active Brake) | `0.00` | `0.00` |
| **`Angry` (Tức giận)** | `90.2%` | Lùi nhẹ về phía sau (`safety_override='full'`) | `-0.12` | `0.00` |
| **`Surprise` (Ngạc nhiên)** | `100.0%` | Xoay một vòng $360^\circ$ ($2\pi\text{ rad}$) tại chỗ rồi tự dừng | `0.00` | `+1.00` |
| **`Fear` (Sợ hãi)** | `46.9%` | Tiến nhẹ về phía trước | `+0.12` | `0.00` |

### Kết Quả Đo Đạc Live Demo Trong Gazebo Fortress (`94.25 s`, `951` mẫu Telemetry @ `10 Hz`)
* **Chuỗi chuyển đổi thực tế (`sample_hold_sec=8.5s`):** `Neutral` (khởi tạo) $\rightarrow$ `Happy` (`23.01s`) $\rightarrow$ `Sad` (`32.43s`) $\rightarrow$ `Neutral` (`40.75s`) $\rightarrow$ `Angry` (`48.74s`) $\rightarrow$ `Surprise` (`57.47s`) $\rightarrow$ `Fear` (`66.41s`), sau đó lặp sang Chu kỳ 2 (`Happy` $\rightarrow$ `Sad` kích hoạt phanh an toàn Front LiDAR tại `0.23 m < 0.25 m` $\rightarrow$ `Neutral`).
* **Quỹ đạo & Góc xoay `Surprise 360°`:** Tổng quãng đường `/odom` đạt `7.050 m` (dịch chuyển ròng `2.301 m`); góc quay tích phân `Surprise` đạt `365.261°` (khóa chốt dừng tại ngưỡng `360°` ở `t = 65.17 s`); độ tin cậy AI trung bình toàn phiên `75.96%` (`857` bản tin `/detected_emotion`); kiểm thử tự động `pytest` đạt `41/41 PASSED`.

---

## 🎙️ Bài Tập Phân Vùng Bản Đồ Nhà Kho (Warehouse Map Segmentation) & Điều Khiển Điều Hướng Bằng Giọng Nói Trực Tiếp Qua Microphone Với Nav2

* **Báo cáo kỹ thuật ngắn 2 trang (PDF):** [`turtlebot4_practice4/docs/bao_cao_voice_navigation.pdf`](turtlebot4_practice4/docs/bao_cao_voice_navigation.pdf)
* **Báo cáo kỹ thuật (Markdown):** [`turtlebot4_practice4/docs/bao_cao_voice_navigation.md`](turtlebot4_practice4/docs/bao_cao_voice_navigation.md)
* **Bản đồ gốc:** [`turtlebot4_practice4/maps/map.pgm`](turtlebot4_practice4/maps/map.pgm) & [`turtlebot4_practice4/maps/map.yaml`](turtlebot4_practice4/maps/map.yaml)
* **Bản đồ phân vùng trực quan:** [`turtlebot4_practice4/maps/map_segmented.png`](turtlebot4_practice4/maps/map_segmented.png) & [`turtlebot4_practice4/maps/map_segmented.pgm`](turtlebot4_practice4/maps/map_segmented.pgm)
* **Dữ liệu các khu vực ngữ nghĩa nhà kho:** [`turtlebot4_practice4/config/semantic_regions.json`](turtlebot4_practice4/config/semantic_regions.json) & [`turtlebot4_practice4/config/semantic_regions.yaml`](turtlebot4_practice4/config/semantic_regions.yaml)

### 1. Bảng 05 Khu Vực Chức Năng Trong Nhà Kho (`warehouse.sdf`)
| Mã khu vực | Tên khu vực nhà kho | Tọa độ Pixel $(u, v)$ | Tọa độ ROS `map` $(x, y, \text{yaw})$ | Các câu lệnh giọng nói qua Microphone tiêu biểu |
| :---: | :--- | :---: | :---: | :--- |
| **`A`** | **Khu nhập hàng** | `(338, 240)` | `(1.80, 13.00, 0.00)` | *"Đi đến khu vực A."*, *"Đi đến khu nhập hàng."* |
| **`B`** | **Khu phân loại** | `(358, 394)` | `(2.80, 5.30, 1.57)` | *"Đi đến khu vực B."*, *"Đi đến khu phân loại."* |
| **`C`** | **Khu xuất hàng** | `(366, 476)` | `(3.20, 1.20, 0.00)` | *"Đi đến khu vực C."*, *"Đi đến khu xuất hàng."* |
| **`D`** | **Khu kệ hàng** | `(342, 670)` | `(2.00, -8.50, -1.57)` | *"Đi đến khu vực D."*, *"Đi đến khu kệ hàng."* |
| **`E`** | **Trạm sạc** | `(270, 500)` | `(-1.60, 0.00, 3.14)` | *"Đi đến khu vực E."*, *"Quay về trạm sạc."* |

### 2. Hướng Dẫn Chạy Từng Phần Hoặc Tích Hợp
```bash
# 1. Phân vùng bản đồ nhà kho .pgm bằng OpenCV (Threshold + Morphology + Distance Transform + Watershed):
./src/turtlebot4_practice4/scripts/run_voice_part1_segment_map.sh

# 2. Kiểm thử phần cứng Microphone trực tiếp & Bộ phân tích câu lệnh nhà kho NLP:
./src/turtlebot4_practice4/scripts/run_voice_part2_speech_test.sh

# 3. Chạy tích hợp toàn bộ (Gazebo + AMCL + Nav2 + RViz2 Semantic Markers + Live Microphone Voice Studio):
./src/turtlebot4_practice4/scripts/run_voice_navigation_all.sh
```

---

## 🖐️ Bài Tập Điều Khiển Hành Vi TurtleBot 4 Bằng Cử Chỉ Tay AI (MediaPipe Gesture Recognizer)

### 1. Kiến trúc các ROS 2 Nodes và Launch Files
* **`webcam_cmdvel_check`** ([`webcam_cmdvel_check.py`](turtlebot4_practice4/turtlebot4_practice4/webcam_cmdvel_check.py)): Kiểm tra nhận lệnh vận tốc qua `/cmd_vel` (tiến, lùi, xoay trái–phải) trong Gazebo và đọc/hiển thị hình ảnh từ Webcam laptop bằng OpenCV.
* **`gesture_recognition_node`** ([`gesture_recognition_node.py`](turtlebot4_practice4/turtlebot4_practice4/gesture_recognition_node.py)):
  * Đọc hình ảnh thời gian thực từ Webcam laptop bằng OpenCV.
  * Sử dụng **MediaPipe Gesture Recognizer** (`gesture_recognizer.task`) kết hợp phân tích động học 21 điểm khớp bàn tay (Hand Landmarks) để phát hiện và phân loại **06 cử chỉ tay**.
  * Hiển thị khung xương bàn tay, tên cử chỉ, độ tin cậy (%) và trạng thái Robot trên cửa sổ OpenCV.
  * Publish kết quả nhận diện đã lọc chống nhiễu lên topic `/detected_gesture` (`std_msgs/msg/String`).
* **`robot_behavior_node`** ([`robot_behavior_node.py`](turtlebot4_practice4/turtlebot4_practice4/robot_behavior_node.py)):
  * Subscribe `/detected_gesture` (`std_msgs/msg/String`) và điều khiển TurtleBot 4 qua topic `/cmd_vel` (`geometry_msgs/msg/Twist`).
  * Máy trạng thái hữu hạn (FSM) yêu cầu xác nhận bằng **Giơ ngón cái (`Thumbs_Up`)** trước khi thực thi hành vi, kèm cơ chế chống nhiễu (Debounce + Edge Trigger + Cooldown), tự động dừng khi nhận **`Open_Palm`** hoặc mất tín hiệu (Watchdog Timeout), và bảo vệ chống va chạm qua LiDAR (`/scan`).

### 2. Bảng Ánh Xạ 06 Cử Chỉ Tay & Hành Vi Robot

| Cử chỉ tay | Nhãn trên `/detected_gesture` | Hành vi của TurtleBot 4 |
| :--- | :--- | :--- |
| **Vẫy tay (Wave)** | `Wave` | Chọn hành vi: **Robot xoay trái–phải để chào lại** |
| **Giơ 1 ngón tay (Pointing Up)** | `Pointing_Up` | Chọn hành vi: **Tiến–lùi 4 lần** |
| **Giơ 2 ngón tay (Victory)** | `Victory` | Chọn hành vi: **Di chuyển theo quỹ đạo tròn** |
| **Giơ ngón cái (Thumbs Up)** | `Thumbs_Up` | **Xác nhận và thực hiện** hành vi đã chọn |
| **Xòe bàn tay (Open Palm)** | `Open_Palm` | **Robot dừng ngay lập tức** (Emergency Stop) |
| **Nắm tay (Closed Fist)** | `Closed_Fist` | **Hủy hành vi hiện tại** và trở về trạng thái chờ (`IDLE`) |

### 3. Hướng Dẫn Chạy Từng Phần Hoặc Tích Hợp

```bash
# Chạy tích hợp toàn bộ (Gazebo + Auto Undock ra vùng rộng + Gesture Node + Behavior Node):
./src/turtlebot4_practice4/scripts/run_gesture_all.sh

# Hoặc chạy riêng từng phần:
# Phần 1 – Khởi chạy Simulation, kiểm tra /cmd_vel và hiển thị Webcam OpenCV:
./src/turtlebot4_practice4/scripts/run_gesture_part1_sim_webcam.sh

# Phần 2 – Nhận diện cử chỉ tay bằng AI (MediaPipe Gesture Recognizer):
./src/turtlebot4_practice4/scripts/run_gesture_part2_recognition.sh

# Phần 3 – Điều khiển hành vi TurtleBot 4 từ topic /detected_gesture:
./src/turtlebot4_practice4/scripts/run_gesture_part3_behavior.sh
```
