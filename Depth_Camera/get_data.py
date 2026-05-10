# ============================================================
# ROS2 기본 라이브러리
# ============================================================
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ============================================================
# 데이터 처리 / 파일 저장 라이브러리
# ============================================================
import json
import csv
import os

# ============================================================
# 종료 처리 / 시간 기록 라이브러리
# ============================================================
import signal
from datetime import datetime

# ============================================================
# ROS2 Topics
# /detections      -> YOLO detection 결과 구독
#
# Core Variables
# SAVE_DIR         -> detection save 경로
# CSV_COLUMNS      -> detection feature columns
# csv_path         -> CSV output file 경로
# jsonl_path       -> JSONL output file 경로
# count            -> total saved detections
#
# 저장된 변수
# - YOLO bbox coordinates
# - confidence / class id
# - object distance estimation
# - robot pose
# - odometry velocity
# - IMU sensor values
# - cmd_vel control command
# ============================================================





# ============================================================
# 저장 경로 / 구독 토픽 설정
# ============================================================
# SAVE_DIR : detection 데이터를 저장할 폴더
# TOPIC    : YOLO 노드에서 publish하는 탐지 결과 토픽
SAVE_DIR = "/home/ubuntu/autonomous_driving/yolo/data"
TOPIC = "/detections"


# ============================================================
# CSV 저장 컬럼 정의
# ============================================================
# /detections 토픽으로 들어오는 JSON 데이터 중
# 학습 데이터 또는 분석용으로 저장할 항목들을 CSV 컬럼으로 정의한다.
CSV_COLUMNS = [
    "timestamp",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "conf", "class",
    "center_m", "median_m",
    "pose_x", "pose_y",
    "linear_x", "angular_z",
    "imu_gyro_z",
    "imu_acc_x", "imu_acc_y", "imu_acc_z",
    "cmd_linear_x", "cmd_angular_z",
]


# ============================================================
# 프로그램 실행 상태 플래그 / Ctrl+C 종료 처리
# ============================================================
# rclpy.spin_once()를 while문으로 돌리기 때문에
# Ctrl+C 입력 시 running 값을 False로 바꿔 안전하게 종료한다.
running = True


def signal_handler(sig, frame):
    global running
    print("\n[INFO] Ctrl+C 감지 → 종료 중...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# DetectionSaverNode
# ============================================================
# 역할:
#   1. /detections 토픽을 구독한다.
#   2. 들어온 탐지 결과 JSON을 파싱한다.
#   3. 탐지 결과를 CSV 파일과 JSONL 파일에 동시에 저장한다.
#
# 저장 형식:
#   - CSV   : 표 형태 분석 / 엑셀 확인 / ML 학습 데이터 구성에 유리
#   - JSONL : 원본 detection 구조를 한 줄씩 저장하기에 유리
# ============================================================
class DetectionSaverNode(Node):
    def __init__(self, csv_path, jsonl_path):
        super().__init__('detection_saver_node')

        # ------------------------------------------------------------
        # 저장 파일 경로 / 누적 저장 개수 초기화
        # ------------------------------------------------------------
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.count = 0

        # ------------------------------------------------------------
        # CSV 파일 초기화
        # ------------------------------------------------------------
        # 실행할 때마다 새로운 CSV 파일을 만들고,
        # 첫 줄에는 컬럼명(header)을 기록한다.
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

        # ------------------------------------------------------------
        # JSONL 파일 초기화
        # ------------------------------------------------------------
        # JSONL은 한 줄에 detection 하나씩 저장하는 방식이다.
        # 기존 내용이 있더라도 실행 시 새 파일로 초기화한다.
        open(self.jsonl_path, "w").close()

        # ------------------------------------------------------------
        # Subscriber 생성
        # ------------------------------------------------------------
        # /detections 토픽을 구독한다.
        # 메시지 타입은 std_msgs/String이고,
        # 실제 내용은 JSON 문자열 형태로 들어온다.
        self.create_subscription(
            String,
            TOPIC,
            self.detection_callback,
            10
        )

        print(f"[INFO] 구독 시작: {TOPIC}")
        print(f"[INFO] CSV  저장: {self.csv_path}")
        print(f"[INFO] JSONL 저장: {self.jsonl_path}")

    # ============================================================
    # /detections 콜백 함수
    # ============================================================
    # YOLO 노드가 publish한 detection JSON 문자열을 받아서
    # CSV와 JSONL에 저장한다.
    #
    # 예상 msg.data 구조:
    # [
    #   {
    #     "bbox": [x1, y1, x2, y2],
    #     "conf": 0.85,
    #     "class": 0,
    #     "pose_x": ...,
    #     "pose_y": ...,
    #     ...
    #   }
    # ]
    # ============================================================
    def detection_callback(self, msg):
        # ------------------------------------------------------------
        # JSON 문자열 파싱
        # ------------------------------------------------------------
        # msg.data는 문자열이므로 json.loads()로 Python list/dict로 변환한다.
        # JSON 형식이 깨져 있으면 저장하지 않고 return한다.
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON 파싱 실패: {e}")
            return

        # ------------------------------------------------------------
        # 저장 시각 생성
        # ------------------------------------------------------------
        # ROS 메시지 stamp가 아니라, 이 saver 노드가 저장한 현재 PC 시간을 기록한다.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # ============================================================
        # CSV 저장 영역
        # ============================================================
        # detection 하나당 CSV 한 줄(row)로 저장한다.
        # 없는 값은 get()의 기본값으로 0.0 또는 -1을 넣는다.
        # ============================================================
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

            for det in detections:
                # bbox 값이 없을 경우 기본값 [0, 0, 0, 0] 사용
                bbox = det.get("bbox", [0, 0, 0, 0])

                # CSV 한 줄에 저장할 데이터 구성
                row = {
                    "timestamp":     timestamp,

                    # YOLO bounding box 좌표
                    "bbox_x1":       bbox[0],
                    "bbox_y1":       bbox[1],
                    "bbox_x2":       bbox[2],
                    "bbox_y2":       bbox[3],

                    # 탐지 confidence / class id
                    "conf":          det.get("conf", 0.0),
                    "class":         det.get("class", -1),

                    # 거리 관련 값
                    # center_m, median_m은 detection을 publish하는 쪽에서 넣어준 경우 저장된다.
                    "center_m":      det.get("center_m", 0.0),
                    "median_m":      det.get("median_m", 0.0),

                    # 로봇 위치
                    "pose_x":        det.get("pose_x", 0.0),
                    "pose_y":        det.get("pose_y", 0.0),

                    # odom 기반 속도
                    "linear_x":      det.get("linear_x", 0.0),
                    "angular_z":     det.get("angular_z", 0.0),

                    # IMU 데이터
                    "imu_gyro_z":    det.get("imu_gyro_z", 0.0),
                    "imu_acc_x":     det.get("imu_acc_x", 0.0),
                    "imu_acc_y":     det.get("imu_acc_y", 0.0),
                    "imu_acc_z":     det.get("imu_acc_z", 0.0),

                    # cmd_vel 명령값
                    "cmd_linear_x":  det.get("cmd_linear_x", 0.0),
                    "cmd_angular_z": det.get("cmd_angular_z", 0.0),
                }

                writer.writerow(row)
                self.count += 1

        # ============================================================
        # JSONL 저장 영역
        # ============================================================
        # JSONL은 detection dict 전체를 거의 원본 그대로 저장한다.
        # 단, 저장 시각 timestamp만 추가한다.
        # ============================================================
        with open(self.jsonl_path, "a") as f:
            for det in detections:
                det["timestamp"] = timestamp
                f.write(json.dumps(det, ensure_ascii=False) + "\n")

        # ------------------------------------------------------------
        # 누적 저장 개수 출력
        # ------------------------------------------------------------
        # end="\r"를 사용해서 같은 줄에 계속 갱신되도록 출력한다.
        print(f"[INFO] 누적 저장: {self.count}건", end="\r")

# ============================================================
# main 함수
# ============================================================
# 역할:
#   1. 저장 폴더 생성
#   2. 실행 시각 기반 파일명 생성
#   3. ROS2 노드 실행
#   4. 종료 시 저장 파일 경로 출력
# ============================================================
def main():
    # ------------------------------------------------------------
    # 저장 폴더 생성
    # ------------------------------------------------------------
    # 폴더가 이미 있으면 그대로 사용하고,
    # 없으면 새로 만든다.
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # 실행 시각을 파일명에 포함
    # ------------------------------------------------------------
    # 여러 번 실행해도 파일명이 겹치지 않도록
    # detections_YYYYMMDD_HHMMSS 형식으로 저장한다.
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"detections_{now}.csv")
    jsonl_path = os.path.join(SAVE_DIR, f"detections_{now}.jsonl")

    # ------------------------------------------------------------
    # ROS2 초기화 / 노드 생성
    # ------------------------------------------------------------
    rclpy.init()
    node = DetectionSaverNode(csv_path, jsonl_path)

    try:
        # ------------------------------------------------------------
        # ROS2 spin 루프
        # ------------------------------------------------------------
        # rclpy.spin(node)를 쓰지 않고 spin_once를 사용하는 이유:
        #   Ctrl+C signal_handler에서 running=False가 되었을 때
        #   while문을 빠져나와 저장 경로를 출력하며 종료하기 위함.
        # ------------------------------------------------------------
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.1)

    finally:
        # ------------------------------------------------------------
        # 종료 처리
        # ------------------------------------------------------------
        # 최종 저장 건수와 파일 경로를 출력하고 ROS2를 종료한다.
        print(f"\n[INFO] 최종 저장 건수: {node.count}")
        print(f"[INFO] CSV  → {csv_path}")
        print(f"[INFO] JSONL → {jsonl_path}")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    print(f"[INFO] 구독 시작: {TOPIC}")
    print(f"[INFO] CSV  저장: {self.csv_path}")
    print(f"[INFO] JSONL 저장: {self.jsonl_path}")

    # ============================================================
    # /detections 콜백 함수
    # ============================================================
    # YOLO 노드가 publish한 detection JSON 문자열을 받아서
    # CSV와 JSONL에 저장한다.
    #
    # 예상 msg.data 구조:
    # [
    #   {
    #     "bbox": [x1, y1, x2, y2],
    #     "conf": 0.85,
    #     "class": 0,
    #     "pose_x": ...,
    #     "pose_y": ...,
    #     ...
    #   }
    # ]
    # ============================================================
    def detection_callback(self, msg):
        # ------------------------------------------------------------
        # JSON 문자열 파싱
        # ------------------------------------------------------------
        # msg.data는 문자열이므로 json.loads()로 Python list/dict로 변환한다.
        # JSON 형식이 깨져 있으면 저장하지 않고 return한다.
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON 파싱 실패: {e}")
            return

        # ------------------------------------------------------------
        # 저장 시각 생성
        # ------------------------------------------------------------
        # ROS 메시지 stamp가 아니라, 이 saver 노드가 저장한 현재 PC 시간을 기록한다.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # ============================================================
        # CSV 저장 영역
        # ============================================================
        # detection 하나당 CSV 한 줄(row)로 저장한다.
        # 없는 값은 get()의 기본값으로 0.0 또는 -1을 넣는다.
        # ============================================================
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

            for det in detections:
                # bbox 값이 없을 경우 기본값 [0, 0, 0, 0] 사용
                bbox = det.get("bbox", [0, 0, 0, 0])

                # CSV 한 줄에 저장할 데이터 구성
                row = {
                    "timestamp":     timestamp,

                    # YOLO bounding box 좌표
                    "bbox_x1":       bbox[0],
                    "bbox_y1":       bbox[1],
                    "bbox_x2":       bbox[2],
                    "bbox_y2":       bbox[3],

                    # 탐지 confidence / class id
                    "conf":          det.get("conf", 0.0),
                    "class":         det.get("class", -1),

                    # 거리 관련 값
                    # center_m, median_m은 detection을 publish하는 쪽에서 넣어준 경우 저장된다.
                    "center_m":      det.get("center_m", 0.0),
                    "median_m":      det.get("median_m", 0.0),

                    # 로봇 위치
                    "pose_x":        det.get("pose_x", 0.0),
                    "pose_y":        det.get("pose_y", 0.0),

                    # odom 기반 속도
                    "linear_x":      det.get("linear_x", 0.0),
                    "angular_z":     det.get("angular_z", 0.0),

                    # IMU 데이터
                    "imu_gyro_z":    det.get("imu_gyro_z", 0.0),
                    "imu_acc_x":     det.get("imu_acc_x", 0.0),
                    "imu_acc_y":     det.get("imu_acc_y", 0.0),
                    "imu_acc_z":     det.get("imu_acc_z", 0.0),

                    # cmd_vel 명령값
                    "cmd_linear_x":  det.get("cmd_linear_x", 0.0),
                    "cmd_angular_z": det.get("cmd_angular_z", 0.0),
                }

                writer.writerow(row)
                self.count += 1

        # ============================================================
        # JSONL 저장 영역
        # ============================================================
        # JSONL은 detection dict 전체를 거의 원본 그대로 저장한다.
        # 단, 저장 시각 timestamp만 추가한다.
        # ============================================================
        with open(self.jsonl_path, "a") as f:
            for det in detections:
                det["timestamp"] = timestamp
                f.write(json.dumps(det, ensure_ascii=False) + "\n")

        # ------------------------------------------------------------
        # 누적 저장 개수 출력
        # ------------------------------------------------------------
        # end="\r"를 사용해서 같은 줄에 계속 갱신되도록 출력한다.
        print(f"[INFO] 누적 저장: {self.count}건", end="\r")

# ============================================================
# main 함수
# ============================================================
# 역할:
#   1. 저장 폴더 생성
#   2. 실행 시각 기반 파일명 생성
#   3. ROS2 노드 실행
#   4. 종료 시 저장 파일 경로 출력
# ============================================================
def main():
    # ------------------------------------------------------------
    # 저장 폴더 생성
    # ------------------------------------------------------------
    # 폴더가 이미 있으면 그대로 사용하고,
    # 없으면 새로 만든다.
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # 실행 시각을 파일명에 포함
    # ------------------------------------------------------------
    # 여러 번 실행해도 파일명이 겹치지 않도록
    # detections_YYYYMMDD_HHMMSS 형식으로 저장한다.
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(SAVE_DIR, f"detections_{now}.csv")
    jsonl_path = os.path.join(SAVE_DIR, f"detections_{now}.jsonl")

    # ------------------------------------------------------------
    # ROS2 초기화 / 노드 생성
    # ------------------------------------------------------------
    rclpy.init()
    node = DetectionSaverNode(csv_path, jsonl_path)

    try:
        # ------------------------------------------------------------
        # ROS2 spin 루프
        # ------------------------------------------------------------
        # rclpy.spin(node)를 쓰지 않고 spin_once를 사용하는 이유:
        #   Ctrl+C signal_handler에서 running=False가 되었을 때
        #   while문을 빠져나와 저장 경로를 출력하며 종료하기 위함.
        # ------------------------------------------------------------
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.1)

    finally:
        # ------------------------------------------------------------
        # 종료 처리
        # ------------------------------------------------------------
        # 최종 저장 건수와 파일 경로를 출력하고 ROS2를 종료한다.
        print(f"\n[INFO] 최종 저장 건수: {node.count}")
        print(f"[INFO] CSV  → {csv_path}")
        print(f"[INFO] JSONL → {jsonl_path}")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
