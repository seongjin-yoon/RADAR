# =========================
# 기본 라이브러리 import
# =========================
import json
import math
import time
from typing import Optional


# =========================
# ROS2 관련 import
# =========================
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Twist

import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs  # noqa: F401


# =========================
# ML 거리 예측 모델 관련 import
# =========================
import pandas as pd
import joblib




# ============================================================
# DetectionMarkerFusionNode
# - YOLO detection 결과(/detections)를 받아서
# - LiDAR(/scan) 또는 ML 모델로 대상까지의 거리를 계산하고
# - base_link 기준 좌표를 map 좌표로 TF 변환한 뒤
# - RViz MarkerArray(/detection_markers)와 위험 좌표(/danger_detected)를 publish하는 노드
# ============================================================
class DetectionMarkerFusionNode(Node):
    def __init__(self):
        super().__init__('detection_marker_fusion_node')

        # ------------------------------------------------------------
        # 1. 파라미터 선언부
        # launch 파일이나 ros2 param으로 외부에서 조정 가능한 값들
        # ------------------------------------------------------------
        self.declare_parameter('camera_fov_deg', 60.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('marker_topic', '/detection_markers')
        self.declare_parameter('detection_topic', '/detections')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('turning_topic', '/robot_turning')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('camera_yaw_offset_deg', 0.0)
        self.declare_parameter('scan_search_half_window', 3)
        self.declare_parameter('min_valid_range', 0.10)
        self.declare_parameter('max_valid_range', 8.0)
        self.declare_parameter('range_offset_m', 0.15)
        self.declare_parameter('duplicate_distance_threshold', 0.50)
        self.declare_parameter('pending_match_distance_threshold', 0.45)
        self.declare_parameter('confirm_count', 3)
        self.declare_parameter('turning_angular_threshold', 0.25)
        self.declare_parameter('imu_gyro_threshold', 0.30)
        self.declare_parameter('turning_hold_sec', 0.25)
        self.declare_parameter('post_turn_ignore_sec', 0.20)
        self.declare_parameter('range_source', 'auto')


        # ------------------------------------------------------------
        # 2. 파라미터 값 읽기
        # declare_parameter로 선언한 값을 실제 변수에 저장
        # ------------------------------------------------------------
        self.camera_fov_deg = float(self.get_parameter('camera_fov_deg').value)
        self.image_width = int(self.get_parameter('image_width').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.marker_topic = str(self.get_parameter('marker_topic').value)
        self.detection_topic = str(self.get_parameter('detection_topic').value)
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.turning_topic = str(self.get_parameter('turning_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.camera_yaw_offset_deg = float(self.get_parameter('camera_yaw_offset_deg').value)
        self.scan_search_half_window = int(self.get_parameter('scan_search_half_window').value)
        self.min_valid_range = float(self.get_parameter('min_valid_range').value)
        self.max_valid_range = float(self.get_parameter('max_valid_range').value)
        self.range_offset_m = float(self.get_parameter('range_offset_m').value)
        self.duplicate_distance_threshold = float(self.get_parameter('duplicate_distance_threshold').value)
        self.pending_match_distance_threshold = float(self.get_parameter('pending_match_distance_threshold').value)
        self.confirm_count = int(self.get_parameter('confirm_count').value)
        self.turning_angular_threshold = float(self.get_parameter('turning_angular_threshold').value)
        self.imu_gyro_threshold = float(self.get_parameter('imu_gyro_threshold').value)
        self.turning_hold_sec = float(self.get_parameter('turning_hold_sec').value)
        self.post_turn_ignore_sec = float(self.get_parameter('post_turn_ignore_sec').value)
        self.range_source = str(self.get_parameter('range_source').value).lower().strip()
        if self.range_source not in ('auto', 'lidar', 'ml'):
            self.get_logger().warn(f'알 수 없는 range_source={self.range_source}, auto로 변경')
            self.range_source = 'auto'

        # YOLO class id 기준
        # 0: 군인, 1: 탱크만 마커로 처리
        self.target_class_ids = [0, 1]

        # ------------------------------------------------------------
        # 3. Subscriber 생성부
        # - /detections      : YOLO 탐지 결과(JSON 문자열)
        # - /scan            : LiDAR 거리 데이터
        # - /robot_turning   : 로봇 회전 상태 Bool
        # - /cmd_vel         : 로봇 속도 명령, angular.z로 회전 여부 판단
        # ------------------------------------------------------------
        self.detection_sub = self.create_subscription(String, self.detection_topic, self.detection_callback, 10)
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, scan_qos)
        self.turning_sub = self.create_subscription(Bool, self.turning_topic, self.turning_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10)

        # ------------------------------------------------------------
        # 4. Publisher 생성부
        # - /detection_markers : RViz에 표시할 군인/탱크 마커 배열
        # - /danger_detected   : 확정된 위험 객체의 map 좌표
        # ------------------------------------------------------------
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.danger_pub = self.create_publisher(PointStamped, '/danger_detected', 10)


        # ------------------------------------------------------------
        # 5. TF Listener 설정
        # base_link 좌표계에서 계산한 객체 위치를 map 좌표계로 변환하기 위해 사용
        # ------------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        # ------------------------------------------------------------
        # 6. 내부 상태 변수
        # latest_scan       : 가장 최근 LiDAR 스캔 저장
        # markers           : RViz에 누적 표시할 MarkerArray
        # danger_positions  : 이미 확정된 객체 좌표, 중복 마킹 방지용
        # pending_positions : 여러 번 같은 위치가 탐지될 때까지 임시 저장
        # is_turning/cmd_turning : 회전 중 마킹 방지용 상태
        # ------------------------------------------------------------
        self.latest_scan = None
        self.markers = MarkerArray()
        self.marker_id = 0
        self.danger_positions = {0: [], 1: []}
        self.pending_positions = {0: [], 1: []}
        self.is_turning = False
        self.cmd_turning = False
        self.turn_ignore_until = 0.0
        self.last_turn_block_log_time = 0.0
        self.tf_warned = False
        self.scan_warned = False

        # ------------------------------------------------------------
        # 7. ML 거리 예측 모델 로드
        # 모델 로드 성공 시 ML 거리 예측 사용 가능
        # 실패하면 자동으로 LiDAR 거리 방식만 사용
        # ------------------------------------------------------------
        try:
            self.model = joblib.load('/home/ubuntu/autonomous_driving/ml/xgb_y_model.pkl')
            self.feature_cols = joblib.load('/home/ubuntu/autonomous_driving/ml/xgb_feature_y_cols.pkl')
            self.use_ml = True
            self.get_logger().info('ML 모델 로드 성공')
        except Exception as e:
            self.model = None
            self.feature_cols = None
            self.use_ml = False
            self.get_logger().warn(f'ML 모델 로드 실패 → LiDAR 방식 사용: {e}')

        self.get_logger().info(
            'DetectionMarkerFusionNode 시작 '
            f'(range_source={self.range_source}, turn_hold={self.turning_hold_sec:.2f}s, '
            f'post_ignore={self.post_turn_ignore_sec:.2f}s)'
        )

    # ============================================================
    # Callback 영역
    # /scan 토픽 callback: 가장 최근 LiDAR 데이터를 저장한다.
    # ============================================================

    def scan_callback(self, msg):
        self.latest_scan = msg

    # ============================================================
    # 회전 중 마킹 방지 유틸 함수
    # ============================================================

    def now_mono(self) -> float:
        return time.monotonic()

    def clear_pending_positions(self):
        if any(len(v) > 0 for v in self.pending_positions.values()):
            self.pending_positions = {0: [], 1: []}
            self.get_logger().info('[turning] pending 후보 초기화')

    def extend_turn_ignore(self, duration: Optional[float] = None, reason: str = 'turn'):
        if duration is None:
            duration = self.turning_hold_sec
        self.turn_ignore_until = max(self.turn_ignore_until, self.now_mono() + duration)
        self.clear_pending_positions()

    # /robot_turning 토픽 callback: 회전 상태를 받아 마킹을 잠시 차단한다.
    def turning_callback(self, msg: Bool):
        was_turning = self.is_turning
        self.is_turning = bool(msg.data)
        if self.is_turning:
            self.extend_turn_ignore(self.turning_hold_sec, reason='robot_turning_true')
        elif was_turning:
            self.turn_ignore_until = max(
                self.turn_ignore_until,
                self.now_mono() + self.post_turn_ignore_sec,
            )

    # /cmd_vel 토픽 callback: angular.z가 크면 로봇이 회전 중이라고 판단한다.
    def cmd_vel_callback(self, msg: Twist):
        self.cmd_turning = abs(float(msg.angular.z)) > self.turning_angular_threshold
        if self.cmd_turning:
            self.extend_turn_ignore(self.turning_hold_sec, reason='cmd_vel_angular')

    def should_ignore_due_to_turning(self) -> bool:
        return self.is_turning or self.cmd_turning or self.now_mono() < self.turn_ignore_until

    def log_turn_block(self, reason: str):
        now = self.now_mono()
        if now - self.last_turn_block_log_time > 1.0:
            remain = max(0.0, self.turn_ignore_until - now)
            self.get_logger().info(f'[마킹 차단] {reason} remain={remain:.2f}s')
            self.last_turn_block_log_time = now

    def detections_indicate_turning(self, detections) -> bool:
        # 탐지된 객체들을 하나씩 처리
        for det in detections:
            try:
                odom_angular = abs(float(det.get('angular_z', 0.0)))
            except Exception:
                odom_angular = 0.0
            try:
                imu_gyro = abs(float(det.get('imu_gyro_z', 0.0)))
            except Exception:
                imu_gyro = 0.0
            if odom_angular > self.turning_angular_threshold or imu_gyro > self.imu_gyro_threshold:
                self.extend_turn_ignore(self.turning_hold_sec, reason='det_motion')
                self.log_turn_block(f'detection motion angular={odom_angular:.3f}, gyro={imu_gyro:.3f}')
                return True
        return False

    # 
    #   /detections 토픽 callback.
    #   YOLO 탐지 결과를 받아 bbox 중심각 계산 → 거리 계산 → map 좌표 변환 →
    #   pending 누적 → confirm_count 이상이면 Marker와 danger 좌표를 publish한다.
    #
    def detection_callback(self, msg):
        # 회전 중이면 bbox 위치가 흔들리기 때문에 마킹하지 않고 return
        if self.should_ignore_due_to_turning():
            self.clear_pending_positions()
            self.log_turn_block('robot/cmd turning')
            return

        # LiDAR 거리 계산을 위해 최신 /scan 데이터가 필요함
        if self.latest_scan is None:
            if not self.scan_warned:
                self.get_logger().warn('scan 없음 - /scan 토픽 확인 필요')
                self.scan_warned = True
            return

        # /detections는 JSON 문자열로 들어오기 때문에 list 형태로 파싱
        try:
            detections = json.loads(msg.data)
            if not isinstance(detections, list):
                return
        except Exception:
            return

        if self.detections_indicate_turning(detections):
            return

        for det in detections:
            try:
                bbox = det['bbox']
                cls = int(det.get('class', -1))
            except Exception:
                continue
            if cls not in self.target_class_ids:
                continue
            if not self._is_valid_bbox(bbox):
                continue

            # bbox 중심 x좌표를 카메라 기준 수평 각도로 변환
            angle_cam = self.bbox_center_to_angle_rad(bbox, self.image_width, self.camera_fov_deg)
            angle_base = angle_cam + math.radians(self.camera_yaw_offset_deg)

            # 객체 방향(angle_base)에 해당하는 거리 계산
            # range_source=auto이면 LiDAR 우선, 실패 시 ML 사용
            measured_range, range_source = self.compute_range(det, angle_base)
            if measured_range is None:
                continue

            measured_range = measured_range - self.range_offset_m
            if measured_range < self.min_valid_range or measured_range > self.max_valid_range:
                continue

            # base_link 좌표계 기준 객체 위치 계산
            base_point = PointStamped()
            base_point.header.frame_id = self.base_frame
            self.apply_detection_stamp(base_point, det)
            base_point.point.x = measured_range * math.cos(angle_base)
            base_point.point.y = measured_range * math.sin(angle_base)
            base_point.point.z = 0.0

            # base_link 기준 좌표를 map 좌표계로 변환
            map_point = self.transform_point_to_map(base_point)
            if map_point is None:
                continue

            map_x = float(map_point.point.x)
            map_y = float(map_point.point.y)

            if self.is_duplicate_position(map_x, map_y, cls):
                continue

            # 같은 위치가 confirm_count만큼 반복 탐지되면 최종 마커로 확정
            self.add_or_update_pending(cls, map_x, map_y, measured_range, range_source)

    # ============================================================
    # 거리 계산 영역
    # LiDAR 우선 또는 ML 모델을 이용해 탐지 객체까지의 거리 계산
    # ============================================================

    def compute_range(self, det, angle_base: float):
        # range_source 설정에 따라 LiDAR 또는 ML로 객체까지의 거리를 계산한다.
        if self.range_source in ('auto', 'lidar'):
            lidar_range = self.find_range_for_angle(
                scan=self.latest_scan,
                target_angle_rad=angle_base,
                half_window=self.scan_search_half_window
            )
            if lidar_range is not None:
                return lidar_range, 'lidar'

        if self.range_source in ('auto', 'ml') and self.use_ml:
            feature_dict = self.make_feature_dict(det)
            if feature_dict is None:
                return None, None
            try:
                X_input = pd.DataFrame([feature_dict], columns=self.feature_cols)
                ml_range = float(self.model.predict(X_input)[0])
                return ml_range, 'ml'
            except Exception as e:
                self.get_logger().warn(f'ML 예측 실패: {e}')
                return None, None

        return None, None

    # ============================================================
    # Detection timestamp 처리
    # YOLO 탐지 시점의 stamp를 PointStamped에 적용
    # ============================================================

    def apply_detection_stamp(self, point: PointStamped, det):
        try:
            sec = int(det.get('stamp_sec', 0))
            nanosec = int(det.get('stamp_nanosec', 0))
        except Exception:
            sec = 0
            nanosec = 0
        point.header.stamp.sec = sec
        point.header.stamp.nanosec = nanosec

    # ============================================================
    # Pending / Confirm 처리 영역
    # 같은 위치가 여러 번 감지될 때만 최종 마커로 확정하여 오탐을 줄임
    # ============================================================

    def add_or_update_pending(self, cls: int, map_x: float, map_y: float, measured_range: float, range_source: str):
        # 탐지 위치를 pending 목록에 누적하고 confirm_count 이상이면 확정 마킹한다.
        found = False
        cls_name = self.class_name(cls)

        for i, (px, py, cnt) in enumerate(self.pending_positions[cls]):
            if math.hypot(map_x - px, map_y - py) < self.pending_match_distance_threshold:
                new_cnt = cnt + 1
                new_px = (px * cnt + map_x) / new_cnt
                new_py = (py * cnt + map_y) / new_cnt
                self.pending_positions[cls][i] = (new_px, new_py, new_cnt)
                found = True
                self.get_logger().info(
                    f'[{cls_name} 누적] ({new_px:.2f}, {new_py:.2f}) '
                    f'cnt={new_cnt}/{self.confirm_count}, range={measured_range:.2f}m/{range_source}'
                )
                if new_cnt >= self.confirm_count:
                    self.confirm_marker(cls, new_px, new_py, measured_range, range_source)
                    self.pending_positions[cls].pop(i)
                break

        if not found:
            self.pending_positions[cls].append((map_x, map_y, 1))
            self.get_logger().info(
                f'[{cls_name} 신규] ({map_x:.2f}, {map_y:.2f}) '
                f'cnt=1/{self.confirm_count}, range={measured_range:.2f}m/{range_source}'
            )

    # 확정된 객체를 MarkerArray와 danger_detected 토픽으로 publish한다.
    def confirm_marker(self, cls: int, x: float, y: float, measured_range: float, range_source: str):
        self.danger_positions[cls].append((x, y))
        if cls == 0:
            marker = self.make_green_marker(x, y)
        else:
            marker = self.make_red_marker(x, y)
        self.markers.markers.append(marker)
        
        # RViz에 누적된 MarkerArray publish
        self.marker_pub.publish(self.markers)

        danger_msg = PointStamped()
        danger_msg.header.frame_id = self.map_frame
        danger_msg.header.stamp = self.get_clock().now().to_msg()
        danger_msg.point.x = x
        danger_msg.point.y = y
        danger_msg.point.z = 0.0

        # 다른 노드가 위험 좌표를 사용할 수 있도록 PointStamped publish
        self.danger_pub.publish(danger_msg)
        self.get_logger().info(
            f'[{self.class_name(cls)} 확정 마킹] map=({x:.2f}, {y:.2f}), '
            f'range={measured_range:.2f}m/{range_source}'
        )

    # ============================================================
    # 좌표 / 각도 / 거리 변환 유틸 함수
    # ============================================================

    def class_name(self, cls: int) -> str:
        return '군인' if cls == 0 else '탱크'

    def bbox_center_to_angle_rad(self, bbox, img_w, fov_deg):
        x1, _, x2, _ = bbox
        cx = (x1 + x2) / 2.0
        normalized = (cx - (img_w / 2.0)) / (img_w / 2.0)
        return normalized * math.radians(fov_deg / 2.0)

    def find_range_for_angle(self, scan, target_angle_rad, half_window=3):
        if scan is None or scan.angle_increment == 0.0:
            return None
        center_idx = int(round((target_angle_rad - scan.angle_min) / scan.angle_increment))
        valid_ranges = []
        for i in range(center_idx - half_window, center_idx + half_window + 1):
            if i < 0 or i >= len(scan.ranges):
                continue
            r = scan.ranges[i]
            if math.isinf(r) or math.isnan(r):
                continue
            if r < self.min_valid_range or r > self.max_valid_range:
                continue
            valid_ranges.append(float(r))
        if not valid_ranges:
            return None
        valid_ranges.sort()
        idx = max(0, int(len(valid_ranges) * 0.3) - 1)
        return valid_ranges[idx]

    # ============================================================
    # TF 변환 영역
    # base_link 기준 객체 좌표를 map 좌표계로 변환
    # ============================================================

    def transform_point_to_map(self, point_in_base):
        try:
            return self.tf_buffer.transform(point_in_base, self.map_frame, timeout=Duration(seconds=0.25))
        except TransformException as e:
            if point_in_base.header.stamp.sec != 0 or point_in_base.header.stamp.nanosec != 0:
                try:
                    fallback = PointStamped()
                    fallback.header.frame_id = point_in_base.header.frame_id
                    fallback.header.stamp.sec = 0
                    fallback.header.stamp.nanosec = 0
                    fallback.point = point_in_base.point
                    return self.tf_buffer.transform(fallback, self.map_frame, timeout=Duration(seconds=0.25))
                except Exception:
                    pass
            if not self.tf_warned:
                self.get_logger().warn(f'TF 변환 실패: {e}')
                self.tf_warned = True
            return None
        except Exception as e:
            if not self.tf_warned:
                self.get_logger().warn(f'Point transform 예외: {e}')
                self.tf_warned = True
            return None

    # ============================================================
    # 중복 마킹 방지
    # 이미 확정된 위치와 너무 가까우면 같은 객체로 보고 무시
    # ============================================================

    def is_duplicate_position(self, x, y, cls):
        for px, py in self.danger_positions[cls]:
            if math.hypot(x - px, y - py) < self.duplicate_distance_threshold:
                return True
        return False

    # ============================================================
    # Marker 생성 영역
    # 군인: 초록색 sphere, 탱크: 빨간색 sphere
    # ============================================================

    def make_green_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'soldier_markers'
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.10
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.22
        marker.scale.y = 0.22
        marker.scale.z = 0.22
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.95
        marker.lifetime.sec = 0
        marker.frame_locked = False
        return marker

    def make_red_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'tank_markers'
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.10
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.22
        marker.scale.y = 0.22
        marker.scale.z = 0.22
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.95
        marker.lifetime.sec = 0
        marker.frame_locked = False
        return marker

    # ============================================================
    # bbox 검증 및 ML feature 생성
    # ============================================================

    def _is_valid_bbox(self, bbox):
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return False
        if x1 < 0 or x2 > self.image_width:
            return False
        return True

    def make_feature_dict(self, det):
        try:
            bbox = det['bbox']
            cls = int(det.get('class', -1))
            if not self._is_valid_bbox(bbox):
                return None
            x1, y1, x2, y2 = bbox
            bbox_width = float(x2 - x1)
            bbox_height = float(y2 - y1)
            bbox_center_x = float((x1 + x2) / 2.0)
            bbox_center_y = float((y1 + y2) / 2.0)
            ratio_mean = float(bbox_width / bbox_height) if bbox_height > 0 else 0.0
            return {
                'bbox_width': bbox_width,
                'bbox_height': bbox_height,
                'bbox_center_x': bbox_center_x,
                'ratio_mean': ratio_mean,
                'class': cls,
                'bbox_center_y': bbox_center_y,
                'bbox_y1': y1,
                'bbox_y2': y2,
            }
        except Exception as e:
            self.get_logger().warn(f'feature 계산 실패: {e}')
            return None



# ============================================================
# main 함수
# ROS2 노드 초기화 → spin으로 callback 대기 → 종료 처리
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = DetectionMarkerFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
