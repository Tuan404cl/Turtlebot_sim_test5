#!/usr/bin/env bash
# ==============================================================================
# Script chạy tích hợp hệ thống Điều khiển TurtleBot 4 bằng Cảm xúc Khuôn mặt AI
# (Phần 1: Gazebo Sim + Load Map & AMCL + Auto Undock -> Phần 2: DeepFace -> Phần 3: Control)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$*" != *"launch_sim:=false"* ]]; then
    "$SCRIPT_DIR/clean_ros.sh" > /dev/null 2>&1 || true
fi

source /opt/ros/humble/setup.bash
if [ -f "/home/tuan/turtelbot4_ws/install/setup.bash" ]; then
    source /home/tuan/turtelbot4_ws/install/setup.bash
fi

export FASTRTPS_DEFAULT_PROFILES_FILE="/home/tuan/turtelbot4_ws/src/turtlebot4_practice4/config/fastdds_udp_only.xml"
MAP_FILE="/home/tuan/turtelbot4_ws/src/turtlebot4_practice4/maps/map.yaml"

echo "======================================================================"
echo "   TÍCH HỢP: ĐIỀU KHIỂN TURTLEBOT 4 BẰNG CẢM XÚC KHUÔN MẶT AI       "
echo "======================================================================"
echo " Bản đồ nạp trên RViz2: $MAP_FILE"
echo " 1. Khởi chạy Gazebo Simulation (world: warehouse)"
echo " 2. Nạp bản đồ đã lưu (map.yaml) + AMCL Localization + Mở RViz2 (sau 10s)"
echo " 3. Tự động Undock và di chuyển robot ra khu vực rộng thoáng (sau 16s)"
echo " 4. Khởi chạy emotion_control_node (sau 25s)"
echo " 5. Khởi chạy emotion_recognition_node + Cửa sổ Webcam AI (sau 27s)"
echo " (Truyền launch_sim:=false để chỉ chạy ngay 2 node AI & điều khiển)"
echo "----------------------------------------------------------------------"
echo " Bảng ánh xạ 6 cảm xúc -> Chuyển động của TurtleBot 4:"
echo "  - Happy    (Vui vẻ)     : Di chuyển đường cong sang TRÁI (v=+0.18, w=+0.55)"
echo "  - Sad      (Buồn)       : Di chuyển đường cong sang PHẢI (v=+0.18, w=-0.55)"
echo "  - Neutral  (Bình thường): DỪNG tại chỗ                   (v= 0.00, w= 0.00)"
echo "  - Angry    (Tức giận)   : Đi LÙI nhẹ                     (v=-0.12, w= 0.00)"
echo "  - Surprise (Ngạc nhiên) : XOAY TRÒN 1 vòng 360° tại chỗ  (v= 0.00, w=+1.00)"
echo "  - Fear     (Sợ hãi)     : Đi TIẾN nhẹ                    (v=+0.12, w= 0.00)"
echo "----------------------------------------------------------------------"
echo " Phím tắt trên cửa sổ Webcam AI:"
echo "  - [SPACE] : Bật/Tắt Dừng khẩn cấp (khóa về Neutral)"
echo "  - [1]-[6] : Giả lập cảm xúc (1:Happy 2:Sad 3:Neutral 4:Angry 5:Surprise 6:Fear)"
echo "  - [0]     : Quay lại nhận diện AI tự động | [q]/[ESC]: Thoát"
echo "======================================================================"

ros2 launch turtlebot4_practice4 sim_emotion_control.launch.py \
    launch_sim:=true \
    map:="$MAP_FILE" \
    world:=warehouse \
    rviz:=true \
    "$@"
