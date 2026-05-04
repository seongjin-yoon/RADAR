import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import Bool

import tf2_ros
from rclpy.duration import Duration

import math
import time
from collections import deque


class ReactivePatrolNode(Node):
    def __init__(self):
        super().__init__('reactive_patrol_node')

        self.safe_distance          = 0.25
        self.side_critical_distance = 0.10
        self.side_hint_distance     = 0.20
        self.side_turn_gain         = 0.18
        self.forward_speed          = 0.09
        self.turn_speed             = 0.5
        self.turning_angular_threshold = 0.20

        self.side_open_distance     = 0.40
        self.side_angle_min         = 75
        self.side_angle_max         = 105

        self.escaping = False
        self.escape_side = None
        self.escape_start_time = 0.0
        self.escape_timeout = 1.5
        self.latest_scan = None

        self.UTURN_IDLE    = 'IDLE'
        self.UTURN_STOP    = 'STOP'
        self.UTURN_ROTATE  = 'ROTATE'
        self.UTURN_ESCAPE  = 'ESCAPE'
        self.UTURN_FORWARD = 'FORWARD'
        self.uturn_state       = self.UTURN_IDLE
        self.uturn_state_start = 0.0
        self.stop_duration     = 3.0
        self.uturn_duration    = 5.0
        self.escape_duration   = 2.0
        self.forward_duration  = 1.0
        self.uturn_turn_speed  = 0.5

        self.danger_trigger_distance = 0.40
        self.uturn_cooldown  = 5.0
        self.last_uturn_time = 0.0

        self.danger_targets = []
        self.triggered_target = None

        self.grid_resolution = 0.20
        self.visited_counts = {}
        self.recent_cells = deque(maxlen=80)
        self.last_record_time = 0.0
        self.record_interval = 0.5
        self.last_hint_time = 0.0
        self.hint_hold_time = 0.5
        self.current_hint = 'FRONT'

        self.danger_blacklist_radius = 0.6
        self.danger_blacklist_penalty = 50
        self.current_state = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, scan_qos)
        self.danger_sub = self.create_subscription(PointStamped, '/danger_detected', self.danger_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.turning_pub = self.create_publisher(Bool, '/robot_turning', 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('ReactivePatrolNode 시작!')

    def scan_callback(self, msg):
        self.latest_scan = msg

    def log_state(self, state):
        if self.current_state != state:
            self.current_state = state
            self.get_logger().info(f'[상태] ▶ {state}')

    def publish_turning(self, is_turning: bool):
        msg = Bool()
        msg.data = is_turning
        self.turning_pub.publish(msg)

    def danger_callback(self, msg: PointStamped):
        dx = msg.point.x
        dy = msg.point.y

        if time.time() - self.last_uturn_time < self.uturn_cooldown:
            self.get_logger().info(f'[감지 무시] 쿨다운 중 ({self.uturn_cooldown - (time.time() - self.last_uturn_time):.1f}초 남음)')
            return

        steps = int(self.danger_blacklist_radius / self.grid_resolution) + 1
        for gx_offset in range(-steps, steps + 1):
            for gy_offset in range(-steps, steps + 1):
                nx = dx + gx_offset * self.grid_resolution
                ny = dy + gy_offset * self.grid_resolution
                if math.hypot(nx - dx, ny - dy) <= self.danger_blacklist_radius:
                    cell = self.world_to_grid(nx, ny)
                    self.visited_counts[cell] = self.danger_blacklist_penalty

        is_dup = any(math.hypot(dx - px, dy - py) < self.danger_trigger_distance for px, py in self.danger_targets)
        if not is_dup:
            self.danger_targets.append((dx, dy))
            self.get_logger().info(f'[위험 좌표 저장] ({dx:.2f}, {dy:.2f}) | 총 {len(self.danger_targets)}개')

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(), timeout=Duration(seconds=0.1))
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return x, y, yaw
        except Exception:
            return None, None, None

    def world_to_grid(self, x, y):
        return (int(math.floor(x / self.grid_resolution)), int(math.floor(y / self.grid_resolution)))

    def record_visited(self, x, y):
        cell = self.world_to_grid(x, y)
        self.visited_counts[cell] = self.visited_counts.get(cell, 0) + 1
        self.recent_cells.append(cell)

    def check_danger_proximity(self, rx, ry):
        if time.time() - self.last_uturn_time < self.uturn_cooldown:
            return False
        for i, (dx, dy) in enumerate(self.danger_targets):
            dist = math.hypot(rx - dx, ry - dy)
            if dist < self.danger_trigger_distance:
                self.triggered_target = i
                self.get_logger().info(f'[위험 근접] 좌표=({dx:.2f}, {dy:.2f}) | 거리={dist:.2f}m → 유턴!')
                return True
        return False

    def score_direction(self, robot_x, robot_y, direction_yaw, steps=10):
        score = 0.0
        for i in range(1, steps + 1):
            nx = robot_x + math.cos(direction_yaw) * self.grid_resolution * i
            ny = robot_y + math.sin(direction_yaw) * self.grid_resolution * i
            cell = self.world_to_grid(nx, ny)
            score += self.visited_counts.get(cell, 0) * (steps - i + 1)
            if cell in self.recent_cells:
                score += 10.0
        return score

    def get_visited_hint(self, rx, ry, yaw):
        if rx is None:
            return None
        if time.time() - self.last_hint_time > self.hint_hold_time:
            front_score = self.score_direction(rx, ry, yaw, steps=10)
            left_score  = self.score_direction(rx, ry, yaw + math.pi / 2, steps=10)
            right_score = self.score_direction(rx, ry, yaw - math.pi / 2, steps=10)
            min_score = min(front_score, left_score, right_score)
            if min_score == front_score:
                self.current_hint = 'FRONT'
            elif min_score == left_score:
                self.current_hint = 'LEFT'
            else:
                self.current_hint = 'RIGHT'
            self.last_hint_time = time.time()
        return self.current_hint

    def get_range(self, scan, angle_min_deg, angle_max_deg, mode='median'):
        ranges = []
        for i, r in enumerate(scan.ranges):
            angle_deg = math.degrees(scan.angle_min + i * scan.angle_increment) % 360
            a_min = angle_min_deg % 360
            a_max = angle_max_deg % 360
            in_range = (a_min <= angle_deg <= a_max) if a_min <= a_max else (angle_deg >= a_min or angle_deg <= a_max)
            if in_range and not math.isinf(r) and not math.isnan(r) and r > 0.05:
                ranges.append(r)
        if not ranges:
            return float('inf')
        ranges.sort()
        if mode == 'min':
            return ranges[0]
        if mode == 'low':
            return ranges[max(0, int(len(ranges) * 0.25))]
        return ranges[len(ranges) // 2]

    def publish_cmd(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def control_loop(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan
        front      = self.get_range(scan, -30, 30,   mode='low')
        left       = self.get_range(scan,  45, 90,   mode='low')
        right      = self.get_range(scan, -90, -45,  mode='low')
        left_side  = self.get_range(scan,  self.side_angle_min,  self.side_angle_max,  mode='median')
        right_side = self.get_range(scan, -self.side_angle_max, -self.side_angle_min, mode='median')

        left_open  = left_side  > self.side_open_distance
        right_open = right_side > self.side_open_distance

        now = time.time()
        rx, ry, ryaw = self.get_robot_pose()

        if now - self.last_record_time > self.record_interval:
            if rx is not None:
                self.record_visited(rx, ry)
            self.last_record_time = now

        # 1순위: 장애물 탈출
        if self.escaping:
            self.publish_turning(True)
            elapsed = now - self.escape_start_time
            if elapsed > self.escape_timeout and front > self.safe_distance:
                self.escaping = False
                self.escape_side = None
                self.get_logger().info('[탈출 완료] 장애물 회피 성공')
            else:
                self.log_state(f'장애물 탈출 중 → {self.escape_side} 회전')
                self.publish_cmd(0.0, self.turn_speed if self.escape_side == 'LEFT' else -self.turn_speed)
                return

        # 2순위: 유턴 상태머신
        if self.uturn_state != self.UTURN_IDLE:
            elapsed = now - self.uturn_state_start

            if self.uturn_state == self.UTURN_STOP:
                self.publish_turning(True)
                self.log_state('유턴 - 정지 중')
                self.publish_cmd(0.0, 0.0)
                if elapsed > self.stop_duration:
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

            elif self.uturn_state == self.UTURN_ESCAPE:
                self.publish_turning(True)
                self.log_state('유턴 - 전진 탈출 중')
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

            elif self.uturn_state == self.UTURN_FORWARD:
                self.publish_turning(True)
                self.log_state('유턴 - 강제 전진 중')
                self.publish_cmd(self.forward_speed, 0.0)
                if elapsed > self.forward_duration:
                    self.uturn_state = self.UTURN_IDLE
                    self.last_uturn_time = now
                    self.publish_turning(False)
                    if self.triggered_target is not None:
                        try:
                            self.danger_targets.pop(self.triggered_target)
                            self.get_logger().info(f'[위험 좌표 제거] 남은 좌표 {len(self.danger_targets)}개')
                        except Exception:
                            pass
                        self.triggered_target = None
                    if rx is not None:
                        for i in range(1, 8):
                            bx = rx + math.cos(ryaw + math.pi) * self.grid_resolution * i
                            by = ry + math.sin(ryaw + math.pi) * self.grid_resolution * i
                            self.visited_counts[self.world_to_grid(bx, by)] = self.danger_blacklist_penalty
                    self.get_logger().info('[유턴 완료] 정상 주행 재개')
                return

        # 3순위: 위험 좌표 근접 → 유턴
        if (rx is not None and self.uturn_state == self.UTURN_IDLE and
                not self.escaping and self.danger_targets and self.check_danger_proximity(rx, ry)):
            self.uturn_state = self.UTURN_STOP
            self.uturn_state_start = now
            self.publish_turning(True)
            self.get_logger().info('[위험 근접] 유턴 시작!')
            return

        # 4순위: 장애물 감지
        if front < self.safe_distance:
            self.publish_turning(True)
            if not self.escaping:
                self.escaping = True
                self.escape_start_time = now
                self.escape_side = 'LEFT' if left > right else 'RIGHT'
                self.get_logger().info(f'[장애물] 탈출 시작 → {self.escape_side}')
            self.log_state(f'장애물 감지 → {self.escape_side} 회전')
            self.publish_cmd(0.0, self.turn_speed if self.escape_side == 'LEFT' else -self.turn_speed)
            return

        # 5순위: 정상 주행
        if front > self.safe_distance:
            hint = self.get_visited_hint(rx, ry, ryaw) if rx is not None else None

            if left < self.side_critical_distance and right < self.side_critical_distance:
                self.publish_turning(False)
                self.log_state('정상주행 - 양쪽 벽 근접 → 직진')
                self.publish_cmd(self.forward_speed * 0.5, 0.0)
            elif left < self.side_critical_distance:
                self.publish_turning(False)
                self.log_state('정상주행 - 왼쪽 벽 근접 → 오른쪽 보정')
                self.publish_cmd(self.forward_speed * 0.8, -self.side_turn_gain)
            elif right < self.side_critical_distance:
                self.publish_turning(False)
                self.log_state('정상주행 - 오른쪽 벽 근접 → 왼쪽 보정')
                self.publish_cmd(self.forward_speed * 0.8, self.side_turn_gain)
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
            elif left_open and not right_open:
                self.log_state(f'정상주행 - 갈림길 왼쪽 열림 → 힌트={hint}')
                if hint == 'LEFT':
                    self.publish_turning(True)
                    self.publish_cmd(self.forward_speed * 0.8, self.side_turn_gain)
                else:
                    self.publish_turning(False)
                    self.publish_cmd(self.forward_speed, 0.0)
            elif right_open and not left_open:
                self.log_state(f'정상주행 - 갈림길 오른쪽 열림 → 힌트={hint}')
                if hint == 'RIGHT':
                    self.publish_turning(True)
                    self.publish_cmd(self.forward_speed * 0.8, -self.side_turn_gain)
                else:
                    self.publish_turning(False)
                    self.publish_cmd(self.forward_speed, 0.0)
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


def main(args=None):
    rclpy.init(args=args)
    node = ReactivePatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_cmd(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
