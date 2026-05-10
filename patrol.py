# =========================
# ROS2 관련 import
# =========================
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Bool

import tf2_ros
from rclpy.duration import Duration

# =========================
# 기본 라이브러리 import
# =========================
import math
import time
from collections import deque

# ============================================================
# ROS2 Topics
# /scan             -> LiDAR sensor data subscribe
# /danger_detected  -> detected danger position subscribe
# /cmd_vel          -> robot velocity command publish
# /robot_turning    -> robot turning state publish
#
# Core Variables
# safe_distance          -> front obstacle threshold
# forward_speed          -> normal linear velocity
# turn_speed             -> rotation angular velocity
# escaping               -> obstacle escape state
# uturn_state            -> U-turn state machine
# danger_targets         -> detected danger positions
# visited_counts         -> explored grid history
# current_hint           -> exploration direction hint
#
# Core Features
# - LiDAR based obstacle avoidance
# - State-machine based U-turn control
# - TF localization (map -> base_link)
# - Grid-based exploration heuristic
# ============================================================


class ReactivePatrolNode(Node):
    # 목적: LiDAR 기반 반응형 주행. 장애물 회피, 유턴 수행.
    # 연결: /scan, /danger_detected 구독. Twist 퍼블리시.
    def __init__(self):
        super().__init__('reactive_patrol_node')

        # ============================================================
        # 1. 주행 / 회피 기본 파라미터
        # ============================================================
        # safe_distance보다 전방 장애물이 가까우면 회피 동작으로 진입한다.
        self.safe_distance          = 0.25  # 설명: 전방 안전 거리. 회피 트리거.

        # 좌우 벽이 매우 가까울 때 보정 주행을 하기 위한 거리 기준이다.
        self.side_critical_distance = 0.10

        # critical보다는 여유 있지만, 한쪽 벽에 가까워졌다고 판단하는 보정 기준이다.
        self.side_hint_distance     = 0.20

        # 좌우 보정 회전량이다. 값이 클수록 벽에서 더 강하게 멀어진다.
        self.side_turn_gain         = 0.18

        # 정상 주행 선속도와 회피 / 유턴 회전속도이다.
        self.forward_speed          = 0.09
        self.turn_speed             = 0.5

        # 이 각속도 이상이면 로봇이 회전 중이라고 판단한다.
        self.turning_angular_threshold = 0.20

        # ============================================================
        # 2. 갈림길 판단 파라미터
        # ============================================================
        # 측면 거리가 이 값보다 크면 해당 방향이 열려 있다고 판단한다.
        self.side_open_distance     = 0.40

        # 측면 판단에 사용할 라이다 각도 범위이다.
        # 예: 왼쪽은 75~105도, 오른쪽은 -105~-75도 범위를 본다.
        self.side_angle_min         = 75
        self.side_angle_max         = 105

        # ============================================================
        # 3. 장애물 회피 상태 변수
        # ============================================================
        # escaping=True이면 현재 장애물 탈출 회전 중이라는 뜻이다.
        self.escaping = False  # 설명: 탈출 상태.
        self.escape_side = None  # 설명: 탈출 방향.
        self.escape_start_time = 0.0  # 설명: 탈출 시작 시간.
        self.escape_timeout = 1.5  # 설명: 탈출 타임아웃.

        # scan_callback에서 최신 LaserScan 데이터를 저장한다.
        self.latest_scan = None

        # ============================================================
        # 4. 위험 좌표 접근 시 유턴 상태머신 파라미터
        # ============================================================
        # 위험 좌표에 가까워졌을 때 STOP → ROTATE → ESCAPE → FORWARD 순서로 유턴한다.
        self.UTURN_IDLE    = 'IDLE'  # 설명: 유턴 상태: 대기.
        self.UTURN_STOP    = 'STOP'  # 설명: 유턴 상태: 정지.
        self.UTURN_ROTATE  = 'ROTATE'  # 설명: 유턴 상태: 회전.
        self.UTURN_ESCAPE  = 'ESCAPE'  # 설명: 유턴 상태: 탈출.
        self.UTURN_FORWARD = 'FORWARD'  # 설명: 유턴 상태: 전방.

        self.uturn_state       = self.UTURN_IDLE
        self.uturn_state_start = 0.0

        # 각 유턴 단계의 유지 시간이다.
        self.stop_duration     = 3.0
        self.uturn_duration    = 5.0
        self.escape_duration   = 2.0
        self.forward_duration  = 1.0
        self.uturn_turn_speed  = 0.5

        # 위험 좌표와 이 거리 이하로 가까워지면 유턴을 시작한다.
        self.danger_trigger_distance = 0.40

        # 유턴 직후 바로 다시 유턴하지 않도록 하는 쿨다운 시간이다.
        self.uturn_cooldown  = 5.0
        self.last_uturn_time = 0.0

        # /danger_detected로 받은 위험 좌표 목록과 현재 유턴을 유발한 좌표 인덱스이다.
        self.danger_targets = []
        self.triggered_target = None

        # ============================================================
        # 5. 방문 위치 기록 / 탐색 힌트 관련 변수
        # ============================================================
        # map 좌표를 grid_resolution 단위의 격자 셀로 변환해서 방문 횟수를 기록한다.
        self.grid_resolution = 0.20
        self.visited_counts = {}

        # 최근 방문 셀은 추가 페널티를 줘서 같은 곳을 반복해서 도는 것을 줄인다.
        self.recent_cells = deque(maxlen=80)

        # 방문 기록은 너무 자주 하지 않도록 record_interval마다 수행한다.
        self.last_record_time = 0.0
        self.record_interval = 0.5

        # 탐색 방향 힌트는 hint_hold_time 동안 유지해서 방향이 너무 자주 바뀌지 않게 한다.
        self.last_hint_time = 0.0
        self.hint_hold_time = 0.5
        self.current_hint = 'FRONT'

        # 위험 좌표 주변은 방문 점수를 높게 넣어 다시 접근하지 않도록 한다.
        self.danger_blacklist_radius = 0.6
        self.danger_blacklist_penalty = 50

        # 로그가 같은 상태에서 반복 출력되지 않도록 현재 상태를 저장한다.
        self.current_state = None

        # ============================================================
        # 6. TF 설정
        # ============================================================
        # map → base_link 변환을 통해 로봇의 현재 위치와 방향을 얻는다.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ============================================================
        # 7. Subscriber 설정
        # ============================================================
        # /scan: 라이다 거리 데이터를 받아 장애물, 벽, 갈림길을 판단한다.
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            scan_qos
        )

        # /danger_detected: 마킹 노드가 publish한 위험 좌표를 받아 유턴 대상 좌표로 저장한다.
        self.danger_sub = self.create_subscription(
            PointStamped,
            '/danger_detected',
            self.danger_callback,
            10
        )

        # ============================================================
        # 8. Publisher 설정
        # ============================================================
        # /cmd_vel: 실제 로봇 주행 명령을 publish한다.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # /robot_turning: 회전 중 여부를 다른 노드에 알려준다.
        # 예: 마커 노드는 이 값을 보고 회전 중 마킹을 차단할 수 있다.
        self.turning_pub = self.create_publisher(Bool, '/robot_turning', 10)

        # ============================================================
        # 9. Timer 설정
        # ============================================================
        # 0.1초마다 control_loop를 실행한다. 즉, 10Hz 제어 루프이다.
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('ReactivePatrolNode 시작!')

    # ================================================================
    # Callback 1. 라이다 데이터 수신
    # ================================================================
    def scan_callback(self, msg):
        # 최신 LaserScan 메시지를 저장해두고 control_loop에서 사용한다.
        self.latest_scan = msg

    # ================================================================
    # 로그 출력 유틸
    # ================================================================
    def log_state(self, state):
        # 같은 상태 로그가 반복 출력되는 것을 막기 위해 상태가 바뀔 때만 출력한다.
        if self.current_state != state:
            self.current_state = state
            self.get_logger().info(f'[상태] ▶ {state}')

    # ================================================================
    # Publisher 1. 회전 중 여부 publish
    # ================================================================
    def publish_turning(self, is_turning: bool):
        # Bool 메시지로 현재 로봇이 회전 중인지 다른 노드에 알려준다.
        msg = Bool()
        msg.data = is_turning
        self.turning_pub.publish(msg)

    # ================================================================
    # Callback 2. 위험 좌표 수신
    # ================================================================
    def danger_callback(self, msg: PointStamped):
        # DetectionMarkerFusionNode에서 확정 마킹한 위험 좌표를 받는다.
        dx = msg.point.x
        dy = msg.point.y

        # 유턴 직후 쿨다운 중이면 새 위험 좌표를 바로 반응하지 않는다.
        if time.time() - self.last_uturn_time < self.uturn_cooldown:
            self.get_logger().info(
                f'[감지 무시] 쿨다운 중 ({self.uturn_cooldown - (time.time() - self.last_uturn_time):.1f}초 남음)'
            )
            return

        # 위험 좌표 주변 셀에 높은 방문 점수를 부여한다.
        # 이렇게 하면 탐색 힌트 계산 시 위험 지역 방향을 피하게 된다.
        steps = int(self.danger_blacklist_radius / self.grid_resolution) + 1
        for gx_offset in range(-steps, steps + 1):
            for gy_offset in range(-steps, steps + 1):
                nx = dx + gx_offset * self.grid_resolution
                ny = dy + gy_offset * self.grid_resolution
                if math.hypot(nx - dx, ny - dy) <= self.danger_blacklist_radius:
                    cell = self.world_to_grid(nx, ny)
                    self.visited_counts[cell] = self.danger_blacklist_penalty

        # 이미 비슷한 위치의 위험 좌표가 있으면 중복 저장하지 않는다.
        is_dup = any(
            math.hypot(dx - px, dy - py) < self.danger_trigger_distance
            for px, py in self.danger_targets
        )
        if not is_dup:
            self.danger_targets.append((dx, dy))
            self.get_logger().info(f'[위험 좌표 저장] ({dx:.2f}, {dy:.2f}) | 총 {len(self.danger_targets)}개')

    # ================================================================
    # TF 유틸. 현재 로봇 위치 얻기
    # ================================================================
    def get_robot_pose(self):
        try:
            # map 좌표계 기준 base_link 위치를 조회한다.
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
            x = t.transform.translation.x
            y = t.transform.translation.y

            # quaternion을 yaw 각도로 변환한다.
            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            return x, y, yaw
        except Exception:
            # TF를 못 받으면 위치 기반 판단은 생략한다.
            return None, None, None

    # ================================================================
    # 방문 기록 유틸
    # ================================================================
    def world_to_grid(self, x, y):
        # 연속적인 map 좌표를 일정 크기의 격자 좌표로 변환한다.
        return (
            int(math.floor(x / self.grid_resolution)),
            int(math.floor(y / self.grid_resolution))
        )

    def record_visited(self, x, y):
        # 현재 로봇 위치 셀의 방문 횟수를 증가시킨다.
        cell = self.world_to_grid(x, y)
        self.visited_counts[cell] = self.visited_counts.get(cell, 0) + 1
        self.recent_cells.append(cell)

    # ================================================================
    # 위험 좌표 근접 판단
    # ================================================================
    def check_danger_proximity(self, rx, ry):
        # 유턴 쿨다운 중이면 위험 좌표 근접 판단을 하지 않는다.
        if time.time() - self.last_uturn_time < self.uturn_cooldown:
            return False

        # 저장된 위험 좌표 중 하나라도 일정 거리 안에 들어오면 유턴 대상이 된다.
        for i, (dx, dy) in enumerate(self.danger_targets):
            dist = math.hypot(rx - dx, ry - dy)
            if dist < self.danger_trigger_distance:
                self.triggered_target = i
                self.get_logger().info(
                    f'[위험 근접] 좌표=({dx:.2f}, {dy:.2f}) | 거리={dist:.2f}m → 유턴!'
                )
                return True
        return False

    # ================================================================
    # 탐색 방향 점수 계산
    # ================================================================
    def score_direction(self, robot_x, robot_y, direction_yaw, steps=10):
        # 특정 방향으로 몇 칸 앞을 가정하고 방문 점수를 계산한다.
        # 점수가 낮을수록 덜 방문한 방향이므로 탐색 우선순위가 높다.
        score = 0.0
        for i in range(1, steps + 1):
            nx = robot_x + math.cos(direction_yaw) * self.grid_resolution * i
            ny = robot_y + math.sin(direction_yaw) * self.grid_resolution * i
            cell = self.world_to_grid(nx, ny)

            # 가까운 셀일수록 더 큰 가중치를 준다.
            score += self.visited_counts.get(cell, 0) * (steps - i + 1)

            # 최근 지나간 셀은 추가 페널티를 준다.
            if cell in self.recent_cells:
                score += 10.0
        return score

    def get_visited_hint(self, rx, ry, yaw):
        # 현재 위치를 모르면 탐색 힌트를 만들 수 없다.
        if rx is None:
            return None

        # hint_hold_time이 지나면 FRONT / LEFT / RIGHT 세 방향 점수를 다시 계산한다.
        if time.time() - self.last_hint_time > self.hint_hold_time:
            front_score = self.score_direction(rx, ry, yaw, steps=10)
            left_score  = self.score_direction(rx, ry, yaw + math.pi / 2, steps=10)
            right_score = self.score_direction(rx, ry, yaw - math.pi / 2, steps=10)

            # 가장 점수가 낮은 방향, 즉 덜 방문한 방향을 선택한다.
            min_score = min(front_score, left_score, right_score)
            if min_score == front_score:
                self.current_hint = 'FRONT'
            elif min_score == left_score:
                self.current_hint = 'LEFT'
            else:
                self.current_hint = 'RIGHT'

            self.last_hint_time = time.time()
        return self.current_hint

    # ================================================================
    # 라이다 거리 추출 유틸
    # ================================================================
    def get_range(self, scan, angle_min_deg, angle_max_deg, mode='median'):
        # LaserScan 전체 값 중 지정한 각도 범위에 포함되는 유효 거리만 추출한다.
        ranges = []
        for i, r in enumerate(scan.ranges):
            angle_deg = math.degrees(scan.angle_min + i * scan.angle_increment) % 360
            a_min = angle_min_deg % 360
            a_max = angle_max_deg % 360

            # 각도 범위가 0도를 가로지르는 경우도 처리한다.
            in_range = (
                (a_min <= angle_deg <= a_max)
                if a_min <= a_max
                else (angle_deg >= a_min or angle_deg <= a_max)
            )

            # inf, nan, 너무 가까운 노이즈 값은 제외한다.
            if in_range and not math.isinf(r) and not math.isnan(r) and r > 0.05:
                ranges.append(r)

        # 유효한 거리값이 없으면 장애물이 없다고 보고 inf를 반환한다.
        if not ranges:
            return float('inf')

        ranges.sort()

        # min: 가장 가까운 값
        if mode == 'min':
            return ranges[0]

        # low: 하위 25% 지점 값이다.
        # median보다 보수적으로 가까운 장애물을 반영하지만 min보다는 노이즈에 덜 민감하다.
        if mode == 'low':
            return ranges[max(0, int(len(ranges) * 0.25))]

        # 기본값은 중앙값이다.
        return ranges[len(ranges) // 2]

    # ================================================================
    # Publisher 2. 주행 명령 publish
    # ================================================================
    def publish_cmd(self, linear, angular):
        # /cmd_vel로 선속도와 각속도를 publish한다.
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    # ================================================================
    # Main Control Loop
    # ================================================================
    def control_loop(self):
        # 라이다 데이터가 아직 없으면 제어를 수행하지 않는다.
        if self.latest_scan is None:
            return

        scan = self.latest_scan

        # ============================================================
        # A. 라이다 구역별 거리 계산
        # ============================================================
        # front: 전방 장애물 판단
        # left/right: 회피 방향 판단
        # left_side/right_side: 갈림길 개방 여부 판단
        front      = self.get_range(scan, -30, 30,   mode='low')
        left       = self.get_range(scan,  45, 90,   mode='low')
        right      = self.get_range(scan, -90, -45,  mode='low')
        left_side  = self.get_range(scan,  self.side_angle_min,  self.side_angle_max,  mode='median')
        right_side = self.get_range(scan, -self.side_angle_max, -self.side_angle_min, mode='median')

        # 측면이 충분히 열려 있으면 갈림길 후보로 판단한다.
        left_open  = left_side  > self.side_open_distance
        right_open = right_side > self.side_open_distance

        now = time.time()

        # 현재 map 기준 로봇 위치와 방향을 얻는다.
        rx, ry, ryaw = self.get_robot_pose()

        # 일정 주기마다 현재 위치를 방문 기록에 저장한다.
        if now - self.last_record_time > self.record_interval:
            if rx is not None:
                self.record_visited(rx, ry)
            self.last_record_time = now

        # ============================================================
        # B. 1순위: 장애물 탈출 중이면 탈출 동작을 계속 수행
        # ============================================================
        if self.escaping:
            self.publish_turning(True)
            elapsed = now - self.escape_start_time

            # 일정 시간이 지났고 전방이 안전해지면 탈출 완료로 본다.
            if elapsed > self.escape_timeout and front > self.safe_distance:
                self.escaping = False
                self.escape_side = None
                self.get_logger().info('[탈출 완료] 장애물 회피 성공')
            else:
                # 아직 탈출 중이면 정지 회전만 수행한다.
                self.log_state(f'장애물 탈출 중 → {self.escape_side} 회전')
                self.publish_cmd(
                    0.0,
                    self.turn_speed if self.escape_side == 'LEFT' else -self.turn_speed
                )
                return

        # ============================================================
        # C. 2순위: 위험 좌표 접근 후 유턴 상태머신
        # ============================================================
        if self.uturn_state != self.UTURN_IDLE:
            elapsed = now - self.uturn_state_start

            # --------------------------------------------------------
            # C-1. STOP: 유턴 전 잠시 정지
            # 이슈 : 초기단계에서 적 발견시 정지하지않고 유턴을하는 현상이 있어 맵에 부딛히는 현상 발생
            # 해결 : 유턴 전 정지 상태를 추가해서 안정하게 유턴 상태 진입을 한다.
            # --------------------------------------------------------
            if self.uturn_state == self.UTURN_STOP:
                self.publish_turning(True)
                self.log_state('유턴 - 정지 중')
                self.publish_cmd(0.0, 0.0)

                if elapsed > self.stop_duration:
                    # 방문 기록 기반 힌트에 따라 회전 방향을 정한다.
                    hint = self.get_visited_hint(rx, ry, ryaw) if rx is not None else None
                    if hint == 'RIGHT':
                        self.uturn_turn_speed = -self.turn_speed
                        self.get_logger().info('[유턴 방향] visited 힌트 → 오른쪽')
                    else:
                        self.uturn_turn_speed = self.turn_speed
                        self.get_logger().info('[유턴 방향] visited 힌트 → 왼쪽')

                    self.uturn_state = self.UTURN_ROTATE
                    self.uturn_state_start = now
                    self.get_logger().info('[유턴] 180도 회전 시작')
                return

            # --------------------------------------------------------
            # C-2. ROTATE: 제자리 회전
            # --------------------------------------------------------
            elif self.uturn_state == self.UTURN_ROTATE:
                self.publish_turning(True)
                self.log_state(f'유턴 - 180도 회전 중 ({"왼쪽" if self.uturn_turn_speed > 0 else "오른쪽"})')
                self.publish_cmd(0.0, self.uturn_turn_speed)

                if elapsed > self.uturn_duration:
                    self.uturn_state = self.UTURN_ESCAPE
                    self.uturn_state_start = now
                    self.last_hint_time = 0.0
                    self.get_logger().info('[유턴] 전진 탈출 시작')
                return

            # --------------------------------------------------------
            # C-3. ESCAPE: 유턴 후 전진 탈출
            # --------------------------------------------------------
            elif self.uturn_state == self.UTURN_ESCAPE:
                self.publish_turning(True)
                self.log_state('유턴 - 전진 탈출 중')

                # 전진 탈출 중 전방 장애물이 가까우면 일반 장애물 회피로 전환한다.
                if front < self.safe_distance:
                    self.uturn_state = self.UTURN_IDLE
                    self.escaping = True
                    self.escape_start_time = now
                    self.escape_side = 'LEFT' if left > right else 'RIGHT'
                    return

                self.publish_cmd(self.forward_speed, 0.0)

                if elapsed > self.escape_duration:
                    self.uturn_state = self.UTURN_FORWARD
                    self.uturn_state_start = now
                    self.get_logger().info('[유턴] 1초 강제 전진 시작')
                return

            # --------------------------------------------------------
            # C-4. FORWARD: 유턴 완료 후 짧은 강제 전진
            # --------------------------------------------------------
            elif self.uturn_state == self.UTURN_FORWARD:
                self.publish_turning(True)
                self.log_state('유턴 - 강제 전진 중')
                self.publish_cmd(self.forward_speed, 0.0)

                if elapsed > self.forward_duration:
                    # 유턴 상태 종료 및 쿨다운 시작
                    self.uturn_state = self.UTURN_IDLE
                    self.last_uturn_time = now
                    self.publish_turning(False)

                    # 유턴을 유발한 위험 좌표는 목록에서 제거한다.
                    if self.triggered_target is not None:
                        try:
                            self.danger_targets.pop(self.triggered_target)
                            self.get_logger().info(f'[위험 좌표 제거] 남은 좌표 {len(self.danger_targets)}개')
                        except Exception:
                            pass
                        self.triggered_target = None

                    # 방금 되돌아온 방향 뒤쪽에 페널티를 줘서 다시 위험 방향으로 가지 않게 한다.
                    if rx is not None:
                        for i in range(1, 8):
                            bx = rx + math.cos(ryaw + math.pi) * self.grid_resolution * i
                            by = ry + math.sin(ryaw + math.pi) * self.grid_resolution * i
                            self.visited_counts[self.world_to_grid(bx, by)] = self.danger_blacklist_penalty

                    self.get_logger().info('[유턴 완료] 정상 주행 재개')
                return

        # ============================================================
        # D. 3순위: 위험 좌표 근접 → 유턴 시작
        # ============================================================
        if (
            rx is not None
            and self.uturn_state == self.UTURN_IDLE
            and not self.escaping
            and self.danger_targets
            and self.check_danger_proximity(rx, ry)
        ):
            self.uturn_state = self.UTURN_STOP
            self.uturn_state_start = now
            self.publish_turning(True)
            self.get_logger().info('[위험 근접] 유턴 시작!')
            return

        # ============================================================
        # E. 4순위: 전방 장애물 감지 → 회피 시작
        # ============================================================
        if front < self.safe_distance:
            self.publish_turning(True)

            # 새로 회피를 시작하는 경우, 더 넓은 쪽으로 회전 방향을 정한다.
            if not self.escaping:
                self.escaping = True
                self.escape_start_time = now
                self.escape_side = 'LEFT' if left > right else 'RIGHT'
                self.get_logger().info(f'[장애물] 탈출 시작 → {self.escape_side}')

            self.log_state(f'장애물 감지 → {self.escape_side} 회전')
            self.publish_cmd(
                0.0,
                self.turn_speed if self.escape_side == 'LEFT' else -self.turn_speed
            )
            return

        # ============================================================
        # F. 5순위: 정상 주행 / 벽 보정 / 갈림길 선택
        # ============================================================
        if front > self.safe_distance:
            # 방문 기록 기반으로 덜 방문한 방향 힌트를 받는다.
            hint = self.get_visited_hint(rx, ry, ryaw) if rx is not None else None

            # --------------------------------------------------------
            # F-1. 양쪽 벽이 모두 가까우면 천천히 직진
            # --------------------------------------------------------
            if left < self.side_critical_distance and right < self.side_critical_distance:
                self.publish_turning(False)
                self.log_state('정상주행 - 양쪽 벽 근접 → 직진')
                self.publish_cmd(self.forward_speed * 0.5, 0.0)

            # --------------------------------------------------------
            # F-2. 왼쪽 벽이 가까우면 오른쪽으로 보정
            # --------------------------------------------------------
            elif left < self.side_critical_distance:
                self.publish_turning(False)
                self.log_state('정상주행 - 왼쪽 벽 근접 → 오른쪽 보정')
                self.publish_cmd(self.forward_speed * 0.8, -self.side_turn_gain)

            # --------------------------------------------------------
            # F-3. 오른쪽 벽이 가까우면 왼쪽으로 보정
            # --------------------------------------------------------
            elif right < self.side_critical_distance:
                self.publish_turning(False)
                self.log_state('정상주행 - 오른쪽 벽 근접 → 왼쪽 보정')
                self.publish_cmd(self.forward_speed * 0.8, self.side_turn_gain)

            # --------------------------------------------------------
            # F-4. 양쪽 갈림길이 모두 열려 있는 경우
            # --------------------------------------------------------
            elif left_open and right_open:
                self.log_state(f'정상주행 - 갈림길 양쪽 열림 → 힌트={hint}')
                if hint == 'LEFT':
                    self.publish_turning(True)
                    self.publish_cmd(self.forward_speed * 0.8, self.side_turn_gain)
                elif hint == 'RIGHT':
                    self.publish_turning(True)
                    self.publish_cmd(self.forward_speed * 0.8, -self.side_turn_gain)
                else:
                    self.publish_turning(False)
                    self.publish_cmd(self.forward_speed, 0.0)

            # --------------------------------------------------------
            # F-5. 왼쪽 갈림길만 열려 있는 경우
            # --------------------------------------------------------
            elif left_open and not right_open:
                self.log_state(f'정상주행 - 갈림길 왼쪽 열림 → 힌트={hint}')
                if hint == 'LEFT':
                    self.publish_turning(True)
                    self.publish_cmd(self.forward_speed * 0.8, self.side_turn_gain)
                else:
                    self.publish_turning(False)
                    self.publish_cmd(self.forward_speed, 0.0)

            # --------------------------------------------------------
            # F-6. 오른쪽 갈림길만 열려 있는 경우
            # --------------------------------------------------------
            elif right_open and not left_open:
                self.log_state(f'정상주행 - 갈림길 오른쪽 열림 → 힌트={hint}')
                if hint == 'RIGHT':
                    self.publish_turning(True)
                    self.publish_cmd(self.forward_speed * 0.8, -self.side_turn_gain)
                else:
                    self.publish_turning(False)
                    self.publish_cmd(self.forward_speed, 0.0)

            # --------------------------------------------------------
            # F-7. 갈림길이 아니면 벽 보정 또는 직진
            # --------------------------------------------------------
            else:
                if left < self.side_hint_distance and right >= self.side_hint_distance:
                    self.publish_turning(False)
                    self.log_state('정상주행 - 왼쪽 hint → 오른쪽 보정')
                    self.publish_cmd(self.forward_speed * 0.8, -self.side_turn_gain)
                elif right < self.side_hint_distance and left >= self.side_hint_distance:
                    self.publish_turning(False)
                    self.log_state('정상주행 - 오른쪽 hint → 왼쪽 보정')
                    self.publish_cmd(self.forward_speed * 0.8, self.side_turn_gain)
                else:
                    self.publish_turning(False)
                    self.log_state('정상주행 - 직진')
                    self.publish_cmd(self.forward_speed, 0.0)


# ====================================================================
# ROS2 main 함수
# ====================================================================
def main(args=None):
    rclpy.init(args=args)
    node = ReactivePatrolNode()
    try:
        # spin을 돌면서 subscriber callback과 timer callback을 계속 처리한다.
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C로 종료할 때 로봇이 계속 움직이지 않도록 정지 명령을 보낸다.
        node.publish_cmd(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
