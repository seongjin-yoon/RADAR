
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 이 모듈은 ROS 노드로 YOLO 객체 감지를 수행하며, 카메라 이미지를 처리하고 압축 이미지를 퍼블리시.
# 주요 기능: YOLO 모델 로드/추론, 모션 기반 감지 차단, ROS 토픽 퍼블리시.
# 연결: detection_marker_node.py와 연동하여 감지 결과를 융합.

import argparse
import json
import os
import signal
import shutil
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, Imu, CompressedImage
from cv_bridge import CvBridge


running = True


def signal_handler(sig, frame):
    global running
    print("\n[INFO] Ctrl+C 감지 → 종료 중...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


@dataclass
class Config:
    # 목적: YOLO 감지 노드의 모든 설정을 중앙화하여 관리. ROS 토픽, 이미지 크기, YOLO 파라미터 등을 포함.
    # 연결: main()에서 인스턴스화되어 YoloCamNode에 전달됨.
    ros_domain_id: int = 11
    camera_topic: str = "/camera/image_raw"
    odom_topic: str = "/odom"
    imu_topic: str = "/imu"
    detection_topic: str = "/detections"
    turning_topic: str = "/robot_turning"
    plain_compressed_topic: str = "/yolo_camera/compressed"
    debug_compressed_topic: str = "/yolo_debug/compressed"
    raw_debug_topic: str = "/yolo_debug_image"
    image_width: int = 640
    image_height: int = 480
    imgsz: int = 416
    conf: float = 0.4
    yolo_max_hz: float = 5.0
    turning_angular_threshold: float = 0.05
    imu_gyro_threshold: float = 0.08
    turning_hold_sec: float = 0.70
    publish_plain_compressed: bool = True
    publish_debug_compressed: bool = True
    publish_raw_debug_image: bool = False
    compressed_width: int = 320
    compressed_height: int = 180
    jpeg_quality: int = 55
    plain_compressed_hz: float = 10.0
    debug_compressed_hz: float = 5.0
    IMGSZ = 416
    CONF  = 0.6
    IMG_W = 640
    IMG_H = 480
    model_name: str = "kd_1_04_03_01"
    weights_path: str = "/home/ubuntu/autonomous_driving/yolo/weights/depth "
    ncnn_path: str = "/home/ubuntu/autonomous_driving/yolo/ncnn"
    NCNN_PATH = "/home/ubuntu/autonomous_driving/yolo/ncnn"
    os.makedirs(NCNN_PATH, exist_ok=True)


def str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"boolean 값이 아닙니다: {value}")


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[Config, Sequence[str]]:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="YOLO detection + compressed preview publisher for Qt UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ros-domain-id", type=int, default=defaults.ros_domain_id)
    parser.add_argument("--camera-topic", default=defaults.camera_topic)
    parser.add_argument("--odom-topic", default=defaults.odom_topic)
    parser.add_argument("--imu-topic", default=defaults.imu_topic)
    parser.add_argument("--detection-topic", default=defaults.detection_topic)
    parser.add_argument("--turning-topic", default=defaults.turning_topic)
    parser.add_argument("--plain-compressed-topic", default=defaults.plain_compressed_topic)
    parser.add_argument("--debug-compressed-topic", default=defaults.debug_compressed_topic)
    parser.add_argument("--raw-debug-topic", default=defaults.raw_debug_topic)
    parser.add_argument("--image-width", type=int, default=defaults.image_width)
    parser.add_argument("--image-height", type=int, default=defaults.image_height)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--conf", type=float, default=defaults.conf)
    parser.add_argument("--yolo-max-hz", type=float, default=defaults.yolo_max_hz)
    parser.add_argument("--turning-angular-threshold", type=float, default=defaults.turning_angular_threshold)
    parser.add_argument("--imu-gyro-threshold", type=float, default=defaults.imu_gyro_threshold)
    parser.add_argument("--turning-hold-sec", type=float, default=defaults.turning_hold_sec)
    parser.add_argument("--publish-plain-compressed", type=str_to_bool, default=defaults.publish_plain_compressed)
    parser.add_argument("--publish-debug-compressed", type=str_to_bool, default=defaults.publish_debug_compressed)
    parser.add_argument("--publish-raw-debug-image", type=str_to_bool, default=defaults.publish_raw_debug_image)
    parser.add_argument("--compressed-width", type=int, default=defaults.compressed_width)
    parser.add_argument("--compressed-height", type=int, default=defaults.compressed_height)
    parser.add_argument("--jpeg-quality", type=int, default=defaults.jpeg_quality)
    parser.add_argument("--plain-compressed-hz", type=float, default=defaults.plain_compressed_hz)
    parser.add_argument("--debug-compressed-hz", type=float, default=defaults.debug_compressed_hz)
    parser.add_argument("--model-name", default=defaults.model_name)
    parser.add_argument("--weights-path", default=defaults.weights_path)
    parser.add_argument("--ncnn-path", default=defaults.ncnn_path)

    parsed, ros_args = parser.parse_known_args(argv)
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


def resize_for_preview(frame_bgr, width: int, height: int):
    if frame_bgr is None:
        return None
    if frame_bgr.shape[1] == width and frame_bgr.shape[0] == height:
        return frame_bgr
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


class YoloCamNode(Node):
    # 목적: ROS 노드로 카메라 이미지에서 YOLO 객체 감지를 수행하고, 압축 이미지를 퍼블리시.
    # 연결: Config를 받아 구독자/퍼블리셔 설정. odom, imu, frame과 연동하여 모션 기반 감지 제어.
    def __init__(self, cfg: Config):
        super().__init__('yolo_cam_node')
        self.cfg = cfg  # 설명: 설정 객체 저장. 모든 파라미터 접근에 사용.
        self.odom = None  # 설명: 최신 오도메트리 데이터. 모션 감지에 사용.
        self.imu = None  # 설명: 최신 IMU 데이터. 모션 감지에 사용.
        self.frame = None  # 설명: 현재 카메라 프레임 (BGR). YOLO 입력으로 사용.
        self.frame_stamp = None  # 설명: 프레임 타임스탬프. 퍼블리시 시 사용.
        self.frame_id = "camera"  # 설명: 프레임 ID. ROS 메시지에 포함.
        self.frame_seq = 0  # 설명: 프레임 시퀀스 번호. 처리 순서 추적.
        self.last_plain_pub_time = 0.0  # 설명: 마지막 일반 압축 퍼블리시 시간. Hz 제어에 사용.
        self.last_debug_pub_time = 0.0  # 설명: 마지막 디버그 압축 퍼블리시 시간. Hz 제어에 사용.
        self.is_turning = False  # 설명: 로봇 턴 상태. 감지 차단에 사용.
        self.turn_ignore_until = 0.0  # 설명: 턴 무시 타이머. 모션 후 감지 재개 시점.
        self.last_turn_block_log_time = 0.0  # 설명: 마지막 턴 차단 로그 시간. 로그 스팸 방지.
        self.bridge = CvBridge()  # 설명: OpenCV-ROS 이미지 변환 브리지.

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, cfg.camera_topic, self.image_callback, qos)
        self.create_subscription(Odometry, cfg.odom_topic, self.odom_callback, 10)
        self.create_subscription(Imu, cfg.imu_topic, self.imu_callback, 10)
        self.create_subscription(Bool, cfg.turning_topic, self.turning_callback, 10)

        self.detection_pub = self.create_publisher(String, cfg.detection_topic, 10)
        self.raw_debug_pub = self.create_publisher(Image, cfg.raw_debug_topic, 10)
        self.plain_compressed_pub = self.create_publisher(CompressedImage, cfg.plain_compressed_topic, 10)
        self.debug_compressed_pub = self.create_publisher(CompressedImage, cfg.debug_compressed_topic, 10)

        self.get_logger().info(f'camera input: {cfg.camera_topic}')
        self.get_logger().info(f'detection output: {cfg.detection_topic}')
        self.get_logger().info(f'plain compressed output: {cfg.plain_compressed_topic} ({cfg.compressed_width}x{cfg.compressed_height}, {cfg.plain_compressed_hz:.1f}Hz, q={cfg.jpeg_quality})')
        self.get_logger().info(f'yolo bbox compressed output: {cfg.debug_compressed_topic} ({cfg.compressed_width}x{cfg.compressed_height}, {cfg.debug_compressed_hz:.1f}Hz, q={cfg.jpeg_quality})')
        self.get_logger().info(f'raw debug image publish: {cfg.publish_raw_debug_image}')
        self.get_logger().info(f'turning gate: topic={cfg.turning_topic}, angular>{cfg.turning_angular_threshold}, gyro>{cfg.imu_gyro_threshold}, hold={cfg.turning_hold_sec}s')

    def now_mono(self) -> float:
        return time.monotonic()

    def turning_callback(self, msg: Bool):
        self.is_turning = bool(msg.data)
        if self.is_turning:
            self.turn_ignore_until = max(self.turn_ignore_until, self.now_mono() + self.cfg.turning_hold_sec)

    def detection_blocked_by_motion(self) -> bool:
        # 목적: 로봇의 모션(턴)을 감지하여 객체 감지를 차단할지 결정. 안정성 확보를 위해 사용.
        # 입력: 없음 (self.odom, self.imu, self.is_turning 사용).
        # 출력: True면 감지 차단, False면 감지 허용.
        # 연결: self.odom, self.imu, self.is_turning과 연동. 모션 임계값 초과 시 차단.
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

    def log_detection_block(self, reason: str):
        now = self.now_mono()
        if now - self.last_turn_block_log_time > 1.0:
            remain = max(0.0, self.turn_ignore_until - now)
            self.get_logger().info(f'[detections 차단] {reason}, remain={remain:.2f}s')
            self.last_turn_block_log_time = now

    def image_callback(self, msg: Image):
        # 목적: 카메라 이미지 메시지를 수신하여 프레임을 저장하고 압축 이미지를 퍼블리시.
        # 입력: msg (ROS Image 메시지).
        # 출력: 없음 (self.frame 업데이트, 퍼블리시).
        # 연결: self.frame에 저장 → main 루프에서 YOLO 입력으로 사용. 압축 퍼블리시는 Hz 제어.
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'프레임 변환 실패: {e}')
            return
        self.frame = frame_bgr
        self.frame_stamp = msg.header.stamp
        self.frame_id = msg.header.frame_id if msg.header.frame_id else 'camera'
        self.frame_seq += 1
        if self.cfg.publish_plain_compressed:
            now = time.monotonic()
            if now - self.last_plain_pub_time >= 1.0 / self.cfg.plain_compressed_hz:
                self.publish_compressed(self.plain_compressed_pub, frame_bgr, stamp=self.frame_stamp, frame_id=self.frame_id)
                self.last_plain_pub_time = now

    def odom_callback(self, msg: Odometry):
        self.odom = msg

    def imu_callback(self, msg: Imu):
        self.imu = msg

    def publish_compressed(self, publisher, frame_bgr, stamp=None, frame_id=None) -> bool:
        # 목적: BGR 이미지를 JPEG 압축하여 ROS CompressedImage 메시지로 퍼블리시.
        # 입력: publisher (ROS 퍼블리셔), frame_bgr (BGR 이미지), stamp/frame_id (메시지 헤더).
        # 출력: 성공 여부 (bool).
        # 연결: OpenCV imencode 사용. cfg.jpeg_quality로 품질 제어.
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


def load_model(cfg: Config):
    # 목적: YOLO 모델을 로드하거나 PyTorch에서 NCNN으로 변환하여 로드.
    # 입력: cfg (설정 객체, 모델 경로 포함).
    # 출력: YOLO 모델 인스턴스.
    # 연결: cfg.weights_path, cfg.ncnn_path 사용. 변환 시 NCNN 저장.
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


def main(argv: Optional[Sequence[str]] = None):
    # 목적: YOLO 감지 노드를 초기화하고 실행. 모델 로드, ROS 스핀 루프에서 YOLO 추론 수행.
    # 입력: argv (명령줄 인자).
    # 출력: 없음 (노드 실행).
    # 연결: Config 파싱 → 모델 로드 → 노드 생성 → YOLO 루프 (frame, odom, imu 연동).
    global running
    cfg, ros_args = parse_args(argv)
    print("PID:", os.getpid())
    print("[INFO] 기본 설정으로 실행합니다. 옵션 확인은 --help")
    print(f"[INFO] YOLO: {cfg.image_width}x{cfg.image_height}, imgsz={cfg.imgsz}, conf={cfg.conf}, max_hz={cfg.yolo_max_hz}")
    print(f"[INFO] Qt preview: plain={cfg.plain_compressed_topic}, debug={cfg.debug_compressed_topic}, {cfg.compressed_width}x{cfg.compressed_height}, jpeg_quality={cfg.jpeg_quality}")

    os.environ.setdefault("ROS_DOMAIN_ID", str(cfg.ros_domain_id))
    rclpy.init(args=list(ros_args) if ros_args else None)
    node = YoloCamNode(cfg)
    model = load_model(cfg)
    print("[INFO] 시작!")

    last_processed_seq = -1
    last_yolo_time = 0.0

    try:
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.frame is None:
                continue
            if node.frame_seq == last_processed_seq:
                continue
            now = time.monotonic()
            if now - last_yolo_time < 1.0 / cfg.yolo_max_hz:
                continue

            last_processed_seq = node.frame_seq
            last_yolo_time = now

            yolo_image = cv2.resize(node.frame.copy(), (cfg.image_width, cfg.image_height), interpolation=cv2.INTER_AREA)
            results = model(yolo_image, conf=cfg.conf, imgsz=cfg.imgsz, verbose=False)
            boxes = results[0].boxes
            detections = []
            block_detections = False

            if boxes is not None and len(boxes) > 0:
                for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0, min(float(x1), cfg.image_width - 1))
                    y1 = max(0, min(float(y1), cfg.image_height - 1))
                    x2 = max(0, min(float(x2), cfg.image_width - 1))
                    y2 = max(0, min(float(y2), cfg.image_height - 1))
                    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(yolo_image, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
                    cv2.putText(yolo_image, f"cls:{int(cls)} conf:{float(conf):.2f}", (ix1, max(iy1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
                            "pose_x": pose.position.x,
                            "pose_y": pose.position.y,
                            "orientation_x": pose.orientation.x,
                            "orientation_y": pose.orientation.y,
                            "orientation_z": pose.orientation.z,
                            "orientation_w": pose.orientation.w,
                            "linear_x": twist.linear.x,
                            "angular_z": twist.angular.z,
                            "imu_gyro_z": imu.angular_velocity.z,
                            "imu_acc_x": imu.linear_acceleration.x,
                            "imu_acc_y": imu.linear_acceleration.y,
                            "imu_acc_z": imu.linear_acceleration.z,
                        })
                        print(f"[탐지] cls:{int(cls)} conf:{float(conf):.2f} pos:({pose.position.x:.2f},{pose.position.y:.2f})")

            if detections:
                msg = String()
                msg.data = json.dumps(detections)
                node.detection_pub.publish(msg)

            if cfg.publish_debug_compressed:
                now = time.monotonic()
                if now - node.last_debug_pub_time >= 1.0 / cfg.debug_compressed_hz:
                    node.publish_compressed(node.debug_compressed_pub, yolo_image, stamp=node.frame_stamp, frame_id=node.frame_id)
                    node.last_debug_pub_time = now

            if cfg.publish_raw_debug_image:
                img_msg = node.bridge.cv2_to_imgmsg(yolo_image, encoding='bgr8')
                img_msg.header.stamp = node.frame_stamp
                img_msg.header.frame_id = node.frame_id
                node.raw_debug_pub.publish(img_msg)

    finally:
        try:
            node.destroy_node()
        finally:
            rclpy.shutdown()
        print("[INFO] 종료")


if __name__ == "__main__":
    main()
