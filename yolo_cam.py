#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# YOLO 카메라 탐지 + Qt UI용 압축 영상 퍼블리시 노드
# ------------------------------------------------------------
# 주요 기능
# 1. 카메라 이미지(/camera/image_raw)를 Subscribe
# 2. YOLO 모델로 군인/탱크 등 객체 탐지
# 3. 탐지 결과를 JSON 문자열로 /detections에 Publish
# 4. Qt UI 표시용 일반 압축 영상과 YOLO bbox 영상 Publish
# 5. 로봇이 회전 중이면 탐지 결과를 차단하여 오검출 마킹 방지
# ============================================================

# ============================================================
# 기본 Python 표준 라이브러리
# ============================================================
import argparse
import json
import os
import signal
import shutil
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# ============================================================
# OPENCV / YOLO 라이브러리
# ============================================================
import cv2
from ultralytics import YOLO

# =========================
# ROS2 관련 import
# =========================

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, Imu, CompressedImage

# ============================================================
# OpenCV 이미지 <-> ROS Image 변환 라이브러리
# ============================================================
from cv_bridge import CvBridge



# ============================================================
# 전역 실행 플래그 / Ctrl+C 종료 처리
# ------------------------------------------------------------
# rclpy.spin()만 쓰는 구조가 아니라 main while 루프에서 spin_once를
# 돌리기 때문에, Ctrl+C 입력 시 running 값을 False로 바꿔 루프를 종료한다.
# ============================================================

running = True


def signal_handler(sig, frame):
    global running
    print("\n[INFO] Ctrl+C 감지 → 종료 중...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# 실행 설정값 Config
# ------------------------------------------------------------
# argparse 기본값으로 사용되는 설정 클래스.
# 토픽 이름, YOLO 입력 크기, confidence, publish 주기,
# 압축 이미지 크기/품질, 모델 경로 등을 한 곳에서 관리한다.
# ============================================================

@dataclass
class Config:
    # ROS / Topic 설정
    ros_domain_id: int = 11
    camera_topic: str = "/camera/image_raw"
    odom_topic: str = "/odom"
    imu_topic: str = "/imu"
    detection_topic: str = "/detections"
    turning_topic: str = "/robot_turning"

    # Qt UI에서 받을 압축 이미지 토픽
    plain_compressed_topic: str = "/yolo_camera/compressed"
    debug_compressed_topic: str = "/yolo_debug/compressed"
    raw_debug_topic: str = "/yolo_debug_image"

    # YOLO 입력 이미지 설정
    image_width: int = 640
    image_height: int = 480
    imgsz: int = 416
    conf: float = 0.4
    yolo_max_hz: float = 5.0

    # 회전 중 탐지 차단 기준
    turning_angular_threshold: float = 0.05
    imu_gyro_threshold: float = 0.08
    turning_hold_sec: float = 0.70

    # 압축 이미지 publish 설정
    publish_plain_compressed: bool = True
    publish_debug_compressed: bool = True
    publish_raw_debug_image: bool = False
    compressed_width: int = 320
    compressed_height: int = 180
    jpeg_quality: int = 55
    plain_compressed_hz: float = 10.0
    debug_compressed_hz: float = 5.0

    # 기존 코드 호환용 상수
    IMGSZ = 416
    CONF  = 0.6
    IMG_W = 640
    IMG_H = 480

    # YOLO 모델 경로 설정
    model_name: str = "kd_1_04_03_01"
    weights_path: str = "/home/ubuntu/autonomous_driving/yolo/weights/depth "
    ncnn_path: str = "/home/ubuntu/autonomous_driving/yolo/ncnn"
    NCNN_PATH = "/home/ubuntu/autonomous_driving/yolo/ncnn"
    os.makedirs(NCNN_PATH, exist_ok=True)


# ============================================================
# argparse 보조 함수
# ------------------------------------------------------------
# 터미널 인자로 True/False 값을 받을 때 문자열을 bool로 변환한다.
# 예: --publish-debug-compressed false
# ============================================================

def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"boolean 값이 아닙니다: {value}")


# ============================================================
# 실행 인자 파싱
# ------------------------------------------------------------
# python 파일 실행 시 옵션으로 토픽명, YOLO 설정, 압축 이미지 설정 등을
# 바꿀 수 있게 한다.
#
# parse_known_args를 사용하므로 ROS에서 넘기는 인자와 일반 argparse 인자를
# 분리해서 처리할 수 있다.
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[Config, Sequence[str]]:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="YOLO detection + compressed preview publisher for Qt UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ROS topic / domain 옵션
    parser.add_argument("--ros-domain-id", type=int, default=defaults.ros_domain_id)
    parser.add_argument("--camera-topic", default=defaults.camera_topic)
    parser.add_argument("--odom-topic", default=defaults.odom_topic)
    parser.add_argument("--imu-topic", default=defaults.imu_topic)
    parser.add_argument("--detection-topic", default=defaults.detection_topic)
    parser.add_argument("--turning-topic", default=defaults.turning_topic)

    # 압축 영상 publish 토픽 옵션
    parser.add_argument("--plain-compressed-topic", default=defaults.plain_compressed_topic)
    parser.add_argument("--debug-compressed-topic", default=defaults.debug_compressed_topic)
    parser.add_argument("--raw-debug-topic", default=defaults.raw_debug_topic)

    # YOLO 추론 옵션
    parser.add_argument("--image-width", type=int, default=defaults.image_width)
    parser.add_argument("--image-height", type=int, default=defaults.image_height)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--conf", type=float, default=defaults.conf)
    parser.add_argument("--yolo-max-hz", type=float, default=defaults.yolo_max_hz)

    # 회전 중 탐지 차단 옵션
    parser.add_argument("--turning-angular-threshold", type=float, default=defaults.turning_angular_threshold)
    parser.add_argument("--imu-gyro-threshold", type=float, default=defaults.imu_gyro_threshold)
    parser.add_argument("--turning-hold-sec", type=float, default=defaults.turning_hold_sec)

    # Qt UI용 이미지 publish 옵션
    parser.add_argument("--publish-plain-compressed", type=str_to_bool, default=defaults.publish_plain_compressed)
    parser.add_argument("--publish-debug-compressed", type=str_to_bool, default=defaults.publish_debug_compressed)
    parser.add_argument("--publish-raw-debug-image", type=str_to_bool, default=defaults.publish_raw_debug_image)
    parser.add_argument("--compressed-width", type=int, default=defaults.compressed_width)
    parser.add_argument("--compressed-height", type=int, default=defaults.compressed_height)
    parser.add_argument("--jpeg-quality", type=int, default=defaults.jpeg_quality)
    parser.add_argument("--plain-compressed-hz", type=float, default=defaults.plain_compressed_hz)
    parser.add_argument("--debug-compressed-hz", type=float, default=defaults.debug_compressed_hz)

    # 모델 파일 옵션
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--weights-path", default=defaults.weights_path)
    parser.add_argument("--ncnn-path", default=defaults.ncnn_path)

    parsed, ros_args = parser.parse_known_args(argv)

    # 잘못된 값 방지를 위해 범위를 제한해서 Config 생성
    cfg = Config(
        ros_domain_id=parsed.ros_domain_id,
        camera_topic=parsed.camera_topic,
        odom_topic=parsed.odom_topic,
        imu_topic=parsed.imu_topic,
        detection_topic=parsed.detection_topic,
        turning_topic=parsed.turning_topic,
        plain_compressed_topic=parsed.plain_compressed_topic,
        debug_compressed_topic=parsed.debug_compressed_topic,
        raw_debug_topic=parsed.raw_debug_topic,
        image_width=max(1, parsed.image_width),
        image_height=max(1, parsed.image_height),
        imgsz=max(1, parsed.imgsz),
        conf=max(0.0, min(1.0, parsed.conf)),
        yolo_max_hz=max(0.1, parsed.yolo_max_hz),
        turning_angular_threshold=max(0.0, parsed.turning_angular_threshold),
        imu_gyro_threshold=max(0.0, parsed.imu_gyro_threshold),
        turning_hold_sec=max(0.0, parsed.turning_hold_sec),
        publish_plain_compressed=parsed.publish_plain_compressed,
        publish_debug_compressed=parsed.publish_debug_compressed,
        publish_raw_debug_image=parsed.publish_raw_debug_image,
        compressed_width=max(1, parsed.compressed_width),
        compressed_height=max(1, parsed.compressed_height),
        jpeg_quality=max(1, min(100, parsed.jpeg_quality)),
        plain_compressed_hz=max(0.1, parsed.plain_compressed_hz),
        debug_compressed_hz=max(0.1, parsed.debug_compressed_hz),
        model_name=parsed.model_name,
        weights_path=parsed.weights_path,
        ncnn_path=parsed.ncnn_path,
    )
    return cfg, ros_args


# ============================================================
# 이미지 크기 변환 함수
# ------------------------------------------------------------
# Qt UI로 보낼 미리보기 이미지를 압축 전 지정 크기로 줄인다.
# 영상 데이터량을 줄여 통신 지연을 낮추기 위한 용도.
# ============================================================

def resize_for_preview(frame_bgr, width: int, height: int):
    if frame_bgr is None:
        return None
    if frame_bgr.shape[1] == width and frame_bgr.shape[0] == height:
        return frame_bgr
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


# ============================================================
# ROS2 YOLO 카메라 노드
# ------------------------------------------------------------
# Subscribe:
#   - 카메라 원본 이미지: cfg.camera_topic
#   - 오도메트리: cfg.odom_topic
#   - IMU: cfg.imu_topic
#   - 회전 여부: cfg.turning_topic
#
# Publish:
#   - YOLO 탐지 결과 JSON: cfg.detection_topic
#   - 일반 압축 카메라 영상: cfg.plain_compressed_topic
#   - bbox가 그려진 압축 영상: cfg.debug_compressed_topic
#   - 원본 debug Image: cfg.raw_debug_topic
# ============================================================

class YoloCamNode(Node):
    def __init__(self, cfg: Config):
        super().__init__('yolo_cam_node')
        self.cfg = cfg

        # ------------------------------------------------------------
        # 최신 센서 데이터 저장 변수
        # ------------------------------------------------------------
        self.odom = None
        self.imu = None
        self.frame = None
        self.frame_stamp = None
        self.frame_id = "camera"
        self.frame_seq = 0

        # ------------------------------------------------------------
        # Publish 주기 제한용 시간 변수
        # ------------------------------------------------------------
        self.last_plain_pub_time = 0.0
        self.last_debug_pub_time = 0.0

        # ------------------------------------------------------------
        # 회전 중 탐지 차단 상태 변수
        # ------------------------------------------------------------
        self.is_turning = False
        self.turn_ignore_until = 0.0
        self.last_turn_block_log_time = 0.0

        # OpenCV 이미지와 ROS Image 메시지 변환용
        self.bridge = CvBridge()

        # ------------------------------------------------------------
        # Subscriber 설정
        # ------------------------------------------------------------
        # 카메라 영상은 최신 프레임만 중요하므로 depth=1, BEST_EFFORT 사용
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, cfg.camera_topic, self.image_callback, qos)
        self.create_subscription(Odometry, cfg.odom_topic, self.odom_callback, 10)
        self.create_subscription(Imu, cfg.imu_topic, self.imu_callback, 10)
        self.create_subscription(Bool, cfg.turning_topic, self.turning_callback, 10)

        # ------------------------------------------------------------
        # Publisher 설정
        # ------------------------------------------------------------
        self.detection_pub = self.create_publisher(String, cfg.detection_topic, 10)
        self.raw_debug_pub = self.create_publisher(Image, cfg.raw_debug_topic, 10)
        self.plain_compressed_pub = self.create_publisher(CompressedImage, cfg.plain_compressed_topic, 10)
        self.debug_compressed_pub = self.create_publisher(CompressedImage, cfg.debug_compressed_topic, 10)

        # 시작 로그
        self.get_logger().info(f'camera input: {cfg.camera_topic}')
        self.get_logger().info(f'detection output: {cfg.detection_topic}')
        self.get_logger().info(f'plain compressed output: {cfg.plain_compressed_topic} ({cfg.compressed_width}x{cfg.compressed_height}, {cfg.plain_compressed_hz:.1f}Hz, q={cfg.jpeg_quality})')
        self.get_logger().info(f'yolo bbox compressed output: {cfg.debug_compressed_topic} ({cfg.compressed_width}x{cfg.compressed_height}, {cfg.debug_compressed_hz:.1f}Hz, q={cfg.jpeg_quality})')
        self.get_logger().info(f'raw debug image publish: {cfg.publish_raw_debug_image}')
        self.get_logger().info(f'turning gate: topic={cfg.turning_topic}, angular>{cfg.turning_angular_threshold}, gyro>{cfg.imu_gyro_threshold}, hold={cfg.turning_hold_sec}s')

    # ============================================================
    # 시간 유틸 함수
    # ------------------------------------------------------------
    # time.monotonic()은 시스템 시간이 바뀌어도 안정적인 시간 측정에 유리하다.
    # ============================================================

    def now_mono(self) -> float:
        return time.monotonic()

    # ============================================================
    # /robot_turning 콜백
    # ------------------------------------------------------------
    # 주행 노드가 회전 중이라고 알려주면 YOLO 탐지는 잠시 차단한다.
    # 회전 중에는 bbox 위치와 실제 방향이 흔들려 마커 좌표가 튈 수 있기 때문이다.
    # ============================================================

    def turning_callback(self, msg: Bool):
        self.is_turning = bool(msg.data)
        if self.is_turning:
            self.turn_ignore_until = max(self.turn_ignore_until, self.now_mono() + self.cfg.turning_hold_sec)

    # ============================================================
    # 탐지 차단 판단 함수
    # ------------------------------------------------------------
    # 다음 조건이면 /detections publish를 막는다.
    # 1. /robot_turning 값이 True인 경우
    # 2. 회전 후 hold 시간이 아직 남은 경우
    # 3. odom angular.z 또는 imu gyro.z가 기준값보다 큰 경우
    # ============================================================

    def detection_blocked_by_motion(self) -> bool:
        now = self.now_mono()
        if self.is_turning or now < self.turn_ignore_until:
            self.log_detection_block('robot_turning/hold')
            return True

        odom_angular = abs(float(self.odom.twist.twist.angular.z)) if self.odom is not None else 0.0
        imu_gyro = abs(float(self.imu.angular_velocity.z)) if self.imu is not None else 0.0

        if odom_angular > self.cfg.turning_angular_threshold or imu_gyro > self.cfg.imu_gyro_threshold:
            self.turn_ignore_until = max(now + self.cfg.turning_hold_sec, self.turn_ignore_until)
            self.log_detection_block(f'motion angular={odom_angular:.3f}, gyro={imu_gyro:.3f}')
            return True
        return False

    # ============================================================
    # 탐지 차단 로그 함수
    # ------------------------------------------------------------
    # 너무 자주 로그가 찍히지 않도록 1초에 한 번 정도만 출력한다.
    # ============================================================

    def log_detection_block(self, reason: str):
        now = self.now_mono()
        if now - self.last_turn_block_log_time > 1.0:
            remain = max(0.0, self.turn_ignore_until - now)
            self.get_logger().info(f'[detections 차단] {reason}, remain={remain:.2f}s')
            self.last_turn_block_log_time = now

    # ============================================================
    # 카메라 이미지 콜백
    # ------------------------------------------------------------
    # 1. ROS Image 메시지를 OpenCV BGR 이미지로 변환
    # 2. 최신 프레임과 timestamp 저장
    # 3. Qt UI용 일반 압축 카메라 영상을 주기 제한 후 publish
    # ============================================================

    def image_callback(self, msg: Image):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'프레임 변환 실패: {e}')
            return

        self.frame = frame_bgr
        self.frame_stamp = msg.header.stamp
        self.frame_id = msg.header.frame_id if msg.header.frame_id else 'camera'
        self.frame_seq += 1

        # bbox 없는 일반 카메라 압축 영상 publish
        if self.cfg.publish_plain_compressed:
            now = time.monotonic()
            if now - self.last_plain_pub_time >= 1.0 / self.cfg.plain_compressed_hz:
                self.publish_compressed(self.plain_compressed_pub, frame_bgr, stamp=self.frame_stamp, frame_id=self.frame_id)
                self.last_plain_pub_time = now

    # ============================================================
    # 오도메트리 / IMU 콜백
    # ------------------------------------------------------------
    # YOLO 탐지 결과에 로봇 위치와 회전 속도 정보를 함께 넣기 위해
    # 최신 odom, imu 메시지를 저장한다.
    # ============================================================

    def odom_callback(self, msg: Odometry):
        self.odom = msg

    def imu_callback(self, msg: Imu):
        self.imu = msg

    # ============================================================
    # 압축 이미지 Publish 함수
    # ------------------------------------------------------------
    # OpenCV BGR 이미지를 resize → JPEG encode → sensor_msgs/CompressedImage로 publish.
    # Qt UI에서 영상만 빠르게 띄우기 위해 원본 Image보다 데이터량을 줄인다.
    # ============================================================

    def publish_compressed(self, publisher, frame_bgr, stamp=None, frame_id=None) -> bool:
        if frame_bgr is None:
            return False

        preview = resize_for_preview(frame_bgr, self.cfg.compressed_width, self.cfg.compressed_height)
        ok, encoded = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])
        if not ok:
            return False

        msg = CompressedImage()
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id if frame_id else self.frame_id
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        publisher.publish(msg)
        return True


# ============================================================
# YOLO / NCNN 모델 로드 함수
# ------------------------------------------------------------
# 1. 이미 변환된 NCNN 모델이 있으면 바로 로드
# 2. 없으면 .pt 모델을 NCNN 형식으로 export
# 3. 변환된 모델 폴더를 ncnn_path로 이동 후 로드
#
# 터틀봇 같은 저전력 환경에서는 일반 PyTorch 모델보다 NCNN 모델이
# 더 가볍게 동작할 수 있다.
# ============================================================

def load_model(cfg: Config):
    os.makedirs(cfg.ncnn_path, exist_ok=True)
    model_path = os.path.join(cfg.ncnn_path, cfg.model_name + "_ncnn_model")
    pt_path = os.path.join(cfg.weights_path, cfg.model_name + ".pt")

    if os.path.exists(model_path):
        print("[INFO] NCNN 모델 로드")
        return YOLO(model_path, task="detect")

    print("[INFO] NCNN 파일 없음 → 변환 시작")
    base_model = YOLO(pt_path)
    base_model.export(format="ncnn", imgsz=cfg.imgsz, half=False)

    generated_path = os.path.join(cfg.weights_path, cfg.model_name + "_ncnn_model")
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    shutil.move(generated_path, model_path)

    print("[INFO] NCNN 변환 완료")
    return YOLO(model_path, task="detect")


# ============================================================
# main 함수
# ------------------------------------------------------------
# 전체 실행 흐름
# 1. 실행 인자 파싱
# 2. ROS_DOMAIN_ID 설정
# 3. ROS2 노드 생성
# 4. YOLO 모델 로드
# 5. while 루프에서 spin_once + 최신 프레임 YOLO 추론
# 6. 탐지 결과 / 압축 debug 영상 publish
# ============================================================

def main(argv: Optional[Sequence[str]] = None):
    global running

    # ------------------------------------------------------------
    # 설정값 로드 및 실행 정보 출력
    # ------------------------------------------------------------
    cfg, ros_args = parse_args(argv)
    print("PID:", os.getpid())
    print("[INFO] 기본 설정으로 실행합니다. 옵션 확인은 --help")
    print(f"[INFO] YOLO: {cfg.image_width}x{cfg.image_height}, imgsz={cfg.imgsz}, conf={cfg.conf}, max_hz={cfg.yolo_max_hz}")
    print(f"[INFO] Qt preview: plain={cfg.plain_compressed_topic}, debug={cfg.debug_compressed_topic}, {cfg.compressed_width}x{cfg.compressed_height}, jpeg_quality={cfg.jpeg_quality}")

    # ------------------------------------------------------------
    # ROS 초기화 및 노드 / 모델 생성
    # ------------------------------------------------------------
    os.environ.setdefault("ROS_DOMAIN_ID", str(cfg.ros_domain_id))
    rclpy.init(args=list(ros_args) if ros_args else None)
    node = YoloCamNode(cfg)
    model = load_model(cfg)
    print("[INFO] 시작!")

    # ------------------------------------------------------------
    # YOLO 추론 주기 제한 변수
    # ------------------------------------------------------------
    last_processed_seq = -1
    last_yolo_time = 0.0

    try:
        while rclpy.ok() and running:
            # ----------------------------------------------------
            # ROS 콜백 처리
            # ----------------------------------------------------
            rclpy.spin_once(node, timeout_sec=0.01)

            # 아직 카메라 프레임이 없으면 대기
            if node.frame is None:
                continue

            # 이미 처리한 프레임이면 중복 추론 방지
            if node.frame_seq == last_processed_seq:
                continue

            # YOLO 최대 Hz 제한
            now = time.monotonic()
            if now - last_yolo_time < 1.0 / cfg.yolo_max_hz:
                continue

            last_processed_seq = node.frame_seq
            last_yolo_time = now

            # ----------------------------------------------------
            # YOLO 입력 이미지 준비 및 추론
            # ----------------------------------------------------
            yolo_image = cv2.resize(node.frame.copy(), (cfg.image_width, cfg.image_height), interpolation=cv2.INTER_AREA)
            results = model(yolo_image, conf=cfg.conf, imgsz=cfg.imgsz, verbose=False)
            boxes = results[0].boxes

            # /detections에 담을 탐지 결과 리스트
            detections = []
            block_detections = False

            # ----------------------------------------------------
            # YOLO bbox 처리
            # ----------------------------------------------------
            # bbox 좌표를 이미지 범위 안으로 보정하고,
            # debug 이미지에는 사각형과 class/confidence를 그린다.
            # odom/imu가 준비된 경우에만 탐지 결과 JSON에 위치/움직임 정보를 넣는다.
            # ----------------------------------------------------
            if boxes is not None and len(boxes) > 0:
                for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                    x1, y1, x2, y2 = box.tolist()

                    # bbox 좌표가 이미지 밖으로 나가지 않게 제한
                    x1 = max(0, min(float(x1), cfg.image_width - 1))
                    y1 = max(0, min(float(y1), cfg.image_height - 1))
                    x2 = max(0, min(float(x2), cfg.image_width - 1))
                    y2 = max(0, min(float(y2), cfg.image_height - 1))

                    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

                    # debug 영상용 bbox 표시
                    cv2.rectangle(yolo_image, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
                    cv2.putText(
                        yolo_image,
                        f"cls:{int(cls)} conf:{float(conf):.2f}",
                        (ix1, max(iy1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    # ------------------------------------------------
                    # 탐지 결과 JSON 구성
                    # ------------------------------------------------
                    # 다음 detection_marker_fusion_node에서 bbox 중심각, stamp,
                    # odom/imu 정보를 이용해 마커 위치 계산에 활용한다.
                    # ------------------------------------------------
                    if (not block_detections) and node.odom is not None and node.imu is not None:
                        pose = node.odom.pose.pose
                        twist = node.odom.twist.twist
                        imu = node.imu

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "conf": float(conf),
                            "class": int(cls),
                            "stamp_sec": int(node.frame_stamp.sec) if node.frame_stamp is not None else 0,
                            "stamp_nanosec": int(node.frame_stamp.nanosec) if node.frame_stamp is not None else 0,
                            "frame_id": node.frame_id,

                            # odom pose
                            "pose_x": pose.position.x,
                            "pose_y": pose.position.y,
                            "orientation_x": pose.orientation.x,
                            "orientation_y": pose.orientation.y,
                            "orientation_z": pose.orientation.z,
                            "orientation_w": pose.orientation.w,

                            # robot motion
                            "linear_x": twist.linear.x,
                            "angular_z": twist.angular.z,

                            # imu
                            "imu_gyro_z": imu.angular_velocity.z,
                            "imu_acc_x": imu.linear_acceleration.x,
                            "imu_acc_y": imu.linear_acceleration.y,
                            "imu_acc_z": imu.linear_acceleration.z,
                        })

                        print(f"[탐지] cls:{int(cls)} conf:{float(conf):.2f} pos:({pose.position.x:.2f},{pose.position.y:.2f})")

            # ----------------------------------------------------
            # 탐지 결과 Publish
            # ----------------------------------------------------
            # detections 리스트를 JSON 문자열로 변환하여 /detections에 publish.
            # Marker Fusion 노드가 이 데이터를 받아 map 좌표 마킹에 사용한다.
            # ----------------------------------------------------
            if detections:
                msg = String()
                msg.data = json.dumps(detections)
                node.detection_pub.publish(msg)

            # ----------------------------------------------------
            # YOLO bbox가 그려진 압축 debug 영상 Publish
            # ----------------------------------------------------
            if cfg.publish_debug_compressed:
                now = time.monotonic()
                if now - node.last_debug_pub_time >= 1.0 / cfg.debug_compressed_hz:
                    node.publish_compressed(node.debug_compressed_pub, yolo_image, stamp=node.frame_stamp, frame_id=node.frame_id)
                    node.last_debug_pub_time = now

            # ----------------------------------------------------
            # 원본 ROS Image debug 영상 Publish
            # ----------------------------------------------------
            # 압축 영상이 아니라 sensor_msgs/Image 형태로 bbox 영상을 보고 싶을 때 사용.
            # 기본값은 False라서 통신량을 줄인다.
            # ----------------------------------------------------
            if cfg.publish_raw_debug_image:
                img_msg = node.bridge.cv2_to_imgmsg(yolo_image, encoding='bgr8')
                img_msg.header.stamp = node.frame_stamp
                img_msg.header.frame_id = node.frame_id
                node.raw_debug_pub.publish(img_msg)

    finally:
        # ------------------------------------------------------------
        # 종료 처리
        # ------------------------------------------------------------
        try:
            node.destroy_node()
        finally:
            rclpy.shutdown()
        print("[INFO] 종료")


if __name__ == "__main__":
    main()
