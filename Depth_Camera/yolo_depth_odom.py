# ============================================================
# 영상 처리 / 수치 계산 라이브러리
# ============================================================
import cv2
import numpy as np

# ============================================================
# RealSense 카메라 라이브러리
# ============================================================
import pyrealsense2 as rs

# ============================================================
# YOLO 객체 탐지 라이브러리
# ============================================================
from ultralytics import YOLO

# ============================================================
# Python 기본 라이브러리
# ============================================================
import shutil
import os
import signal
import sys
import json

# ============================================================
# ROS2 기본 라이브러리
# ============================================================
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist

# ============================================================
# OpenCV 이미지 <-> ROS Image 변환 라이브러리
# ============================================================
from cv_bridge import CvBridge


# ============================================================
# 메인 처리 흐름
# ============================================================
# 1. ROS callback 처리
# 2. RealSense color/depth frame 수신
# 3. depth → color align
# 4. YOLO 추론 수행
# 5. bbox별 depth 계산
# 6. odom / imu / cmd_vel 정보 결합
# 7. /detections publish -> 데이터 수집
# 8. debug image publish -> 동영상 
# ============================================================





# ============================================================
# 프로그램 실행 상태 플래그 / Ctrl+C 종료 처리
# ============================================================
# while 루프를 직접 돌리는 구조이기 때문에
# Ctrl+C 입력 시 running 값을 False로 바꿔 안전하게 종료한다.
# ============================================================
running = True


def signal_handler(sig, frame):
    global running
    print("\n[INFO] Ctrl+C 감지 → 종료 중...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# YoloDepthNode
# ============================================================
# 역할:
#   1. /odom, /imu, /cmd_vel 토픽을 subscribe한다.
#   2. YOLO 탐지 결과를 /detections 토픽으로 publish한다.
#   3. bbox가 그려진 디버그 이미지를 /yolo_debug_image로 publish한다.
#
# 주의:
#   카메라 프레임과 depth 프레임은 ROS 토픽으로 받지 않고,
#   아래 main 루프에서 RealSense pipeline으로 직접 가져온다.
# ============================================================
class YoloDepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_node')

        # ------------------------------------------------------------
        # 수신 데이터 저장 변수
        # ------------------------------------------------------------
        # 각 콜백에서 최신 메시지를 저장해두고,
        # YOLO detection을 만들 때 함께 JSON에 넣는다.
        self.odom = None
        self.odom = None
        self.imu = None
        self.cmd_vel = None

        # ============================================================
        # Subscriber 영역
        # ============================================================
        # /odom    : 로봇 위치, 방향, 선속도, 각속도 정보
        # /imu     : 자이로 z축, 가속도 x/y/z 정보
        # /cmd_vel : 현재 로봇에게 내려간 이동 명령값
        # ============================================================
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            10
        )
        self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # ============================================================
        # Publisher 영역
        # ============================================================
        # /detections       : YOLO bbox + depth 거리 + odom/imu/cmd 정보를 JSON 문자열로 publish
        # /yolo_debug_image : YOLO bbox와 거리 텍스트가 표시된 OpenCV 이미지를 ROS Image로 publish
        # ============================================================
        self.pub = self.create_publisher(
            String,
            '/detections',
            10
        )

        self.image_pub = self.create_publisher(Image, '/yolo_debug_image', 10)
        self.bridge = CvBridge()

    # ============================================================
    # Callback 영역
    # ============================================================
    # subscribe한 토픽의 최신 메시지를 멤버 변수에 저장한다.
    # 실제 데이터 조합과 publish는 메인 while 루프에서 수행된다.
    # ============================================================
    def odom_callback(self, msg):
        self.odom = msg

    def imu_callback(self, msg):
        self.imu = msg

    def cmd_vel_callback(self, msg):
        self.cmd_vel = msg


# ============================================================
# YOLO / 카메라 해상도 설정
# ============================================================
# IMGSZ : YOLO 추론 입력 크기
# CONF  : confidence threshold
#
# COLOR_W, COLOR_H : RealSense RGB 컬러 이미지 해상도
# DEPTH_W, DEPTH_H : RealSense Depth 이미지 해상도
# ============================================================
IMGSZ = 416
CONF = 0.4

COLOR_W, COLOR_H = 640, 480
DEPTH_W, DEPTH_H = 640, 480


# ============================================================
# YOLO 모델 경로 설정
# ============================================================
# pt 모델이 있으면 최초 실행 시 NCNN 형식으로 변환하고,
# 이후부터는 변환된 NCNN 모델을 바로 로드한다.
# ============================================================
model_name = "box_base_best"

weights_path = "/home/ubuntu/autonomous_driving/yolo/weights"
ncnn_path = "/home/ubuntu/autonomous_driving/yolo/ncnn"
os.makedirs(ncnn_path, exist_ok=True)

model_path = os.path.join(ncnn_path, model_name + "_ncnn_model")
pt_path = os.path.join(weights_path, model_name + ".pt")

print("PID:", os.getpid())
print(f"[INFO] Color 해상도: {COLOR_W}×{COLOR_H}")
print(f"[INFO] Depth 해상도: {DEPTH_W}×{DEPTH_H}  (align 후 → {COLOR_W}×{COLOR_H})")


# ============================================================
# YOLO / NCNN 모델 로드 영역
# ============================================================
# 이미 변환된 NCNN 모델이 있으면 바로 로드한다.
# 없으면 .pt 모델을 NCNN으로 export한 뒤 해당 모델을 로드한다.
# ============================================================
if os.path.exists(model_path):
    print("[INFO] NCNN 모델 로드")
    model = YOLO(model_path, task="detect")
else:
    print("[INFO] NCNN 파일 없음 → 변환 시작")
    base_model = YOLO(pt_path)
    base_model.export(format="ncnn", imgsz=IMGSZ, half=False)

    generated_path = os.path.join(weights_path, model_name + "_ncnn_model")
    target_path = os.path.join(ncnn_path, model_name + "_ncnn_model")
    shutil.move(generated_path, target_path)
    model = YOLO(target_path, task="detect")


# ============================================================
# Depth 거리 계산 함수
# ============================================================
# YOLO bbox 내부에서 거리값을 계산한다.
#
# 반환값:
#   center_m : bbox 중심점의 depth 거리
#   median_m : bbox 영역 전체 depth 중 유효값의 중앙값 거리
#
# center_m은 빠르지만 중심점이 빈 depth일 수 있고,
# median_m은 bbox 영역을 활용하기 때문에 조금 더 안정적인 값으로 쓸 수 있다.
# ============================================================
def get_depth_in_box(depth_frame, depth_raw, x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # bbox 중심 좌표
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # 중심점 거리
    center_m = depth_frame.get_distance(cx, cy)

    # bbox 내부 depth ROI 추출
    roi = depth_raw[y1:y2, x1:x2]
    valid = roi[roi > 0]

    # 유효 depth 값이 없으면 0.0 처리
    if len(valid) == 0:
        median_m = 0.0
    else:
        # raw depth 값에 depth unit을 곱해 meter 단위로 변환
        median_m = float(np.median(valid)) * depth_frame.get_units()

    return center_m, median_m


# ============================================================
# ROS2 노드 초기화
# ============================================================
# YoloDepthNode는 odom/imu/cmd_vel을 수신하고,
# detection/debug image를 publish하는 역할을 담당한다.
# ============================================================
rclpy.init()
node = YoloDepthNode()


# ============================================================
# RealSense Pipeline 초기화
# ============================================================
# ROS 카메라 토픽을 쓰는 것이 아니라,
# pyrealsense2를 통해 RealSense 장치에서 직접 color/depth frame을 가져온다.
# ============================================================
print("Before Pipline")
pipeline = rs.pipeline()
config = rs.config()

# 컬러 스트림과 depth 스트림 활성화
config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, 30)

print("Start Pipline")
pipeline.start(config)

# ============================================================
# Depth Align 설정
# ============================================================
# depth frame을 color frame 기준으로 정렬한다.
# 이렇게 해야 YOLO bbox 좌표와 depth 좌표가 같은 기준으로 맞춰진다.
# ============================================================
print("Real Sense Align")
align = rs.align(rs.stream.color)

print("[INFO] 시작")


# ============================================================
# 메인 루프
# ============================================================
# 전체 흐름:
#   1. ROS 콜백 처리
#   2. RealSense color/depth frame 수신
#   3. depth를 color 기준으로 align
#   4. YOLO 추론
#   5. bbox별 depth 거리 계산
#   6. odom/imu/cmd_vel 정보와 함께 JSON 구성
#   7. /detections publish
#   8. /yolo_debug_image publish
# ============================================================
try:
    while rclpy.ok() and running:
        # ------------------------------------------------------------
        # ROS 콜백 처리
        # ------------------------------------------------------------
        # /odom, /imu, /cmd_vel 콜백이 실행되도록 spin_once 수행
        rclpy.spin_once(node, timeout_sec=0.0)

        # ------------------------------------------------------------
        # RealSense 프레임 수신
        # ------------------------------------------------------------
        # color/depth 프레임이 들어올 때까지 대기한다.
        frames = pipeline.wait_for_frames()

        # ------------------------------------------------------------
        # Depth → Color 좌표계 Align
        # ------------------------------------------------------------
        # align이 실패하면 이번 프레임은 건너뛴다.
        try:
            aligned = align.process(frames)
        except RuntimeError:
            print(f"[WARN] align 실패:", flush=True)
            continue

        print("align 성공")

        # ------------------------------------------------------------
        # Align된 color/depth frame 추출
        # ------------------------------------------------------------
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # RealSense frame을 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())

        # ============================================================
        # YOLO 추론 영역
        # ============================================================
        # color_image를 입력으로 객체 탐지를 수행한다.
        # results[0].boxes 안에 bbox, confidence, class 정보가 들어 있다.
        # ============================================================
        results = model(color_image, conf=CONF, imgsz=IMGSZ, verbose=False)
        boxes = results[0].boxes
        detections = []

        # ------------------------------------------------------------
        # Detection 결과 처리
        # ------------------------------------------------------------
        # bbox별로 depth 거리 계산, 화면 표시, JSON 데이터 구성을 수행한다.
        if boxes is not None and len(boxes) > 0:
            for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                x1, y1, x2, y2 = box.tolist()

                # ------------------------------------------------------------
                # bbox 좌표 보정
                # ------------------------------------------------------------
                # bbox가 이미지 범위를 벗어나지 않도록 clamp 처리한다.
                x1 = max(0, min(x1, COLOR_W - 1))
                y1 = max(0, min(y1, COLOR_H - 1))
                x2 = max(0, min(x2, COLOR_W - 1))
                y2 = max(0, min(y2, COLOR_H - 1))

                # ------------------------------------------------------------
                # bbox 내부 depth 거리 계산
                # ------------------------------------------------------------
                center_m, median_m = get_depth_in_box(
                    depth_frame, depth_raw, x1, y1, x2, y2
                )

                # ------------------------------------------------------------
                # Debug Image에 bbox와 텍스트 표시
                # ------------------------------------------------------------
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(color_image, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

                label = (f"cls:{int(cls)} {float(conf):.2f} "
                         f"c:{center_m:.2f}m med:{median_m:.2f}m")
                cv2.putText(
                    color_image, label,
                    (ix1, max(iy1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # ============================================================
                # Detection JSON 구성 영역
                # ============================================================
                # odom, imu, cmd_vel이 모두 들어온 상태에서만
                # detection 데이터를 구성하여 /detections에 publish한다.
                # ============================================================
                if (
                    node.odom is not None and
                    node.imu is not None and
                    node.cmd_vel is not None
                ):
                    cmd_linear_x = node.cmd_vel.linear.x if node.cmd_vel is not None else 0.0
                    cmd_angular_z = node.cmd_vel.angular.z if node.cmd_vel is not None else 0.0

                    pose = node.odom.pose.pose
                    twist = node.odom.twist.twist
                    imu = node.imu
                    cmd = node.cmd_vel

                    # ------------------------------------------------------------
                    # /detections로 publish할 데이터
                    # ------------------------------------------------------------
                    # bbox, confidence, class, depth 거리,
                    # odom 위치/속도, imu, orientation, cmd_vel 값을 하나로 묶는다.
                    data = {
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf),
                        "class": int(cls),
                        "center_m": center_m,
                        "median_m": median_m,

                        # odom 위치 / 속도
                        "pose_x": pose.position.x,
                        "pose_y": pose.position.y,
                        "linear_x": twist.linear.x,
                        "angular_z": twist.angular.z,

                        # imu 자이로 / 가속도
                        "imu_gyro_z": imu.angular_velocity.z,
                        "imu_acc_x": imu.linear_acceleration.x,
                        "imu_acc_y": imu.linear_acceleration.y,
                        "imu_acc_z": imu.linear_acceleration.z,

                        # odom orientation quaternion
                        "ori_x": pose.orientation.x,
                        "ori_y": pose.orientation.y,
                        "ori_z": pose.orientation.z,
                        "ori_w": pose.orientation.w,

                        # cmd_vel 명령값
                        "cmd_linear_x": cmd.linear.x,
                        "cmd_angular_z": cmd.angular.z,
                    }

                    detections.append(data)
                    print(data)

        # ============================================================
        # /detections Publish 영역
        # ============================================================
        # 탐지 결과가 하나 이상 있을 때만 JSON 문자열로 변환해서 publish한다.
        # ============================================================
        if len(detections) > 0:
            msg = String()
            msg.data = json.dumps(detections)
            node.pub.publish(msg)

        # ------------------------------------------------------------
        # 수신되지 않은 ROS 토픽 확인용 로그
        # ------------------------------------------------------------
        # odom/imu/cmd_vel 중 아직 들어오지 않은 토픽이 있으면 출력한다.
        if node.odom is None:
            print("no odom")
        if node.imu is None:
            print("no imu")
        if node.cmd_vel is None:
            print("no cmd_vel")

        # ============================================================
        # /yolo_debug_image Publish 영역
        # ============================================================
        # bbox와 거리 텍스트가 표시된 color_image를 ROS Image로 변환해 publish한다.
        # ============================================================
        img_msg = node.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
        node.image_pub.publish(img_msg)

# ============================================================
# 종료 처리
# ============================================================
# RealSense pipeline을 정지하고 ROS2를 shutdown한다.
# ============================================================
finally:
    pipeline.stop()
    rclpy.shutdown()
    print("[INFO] 종료")
