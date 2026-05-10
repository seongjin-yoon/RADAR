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
# 파일 / 경로 처리 라이브러리
# ============================================================
import shutil
import os
# ─────────────────────────────────────────────
#  설정값
# ─────────────────────────────────────────────
IMGSZ = 416    # YOLO 내부 추론 해상도 (letterbox), 출력 좌표는 color 해상도 기준
CONF  = 0.4    # confidence threshold

# ── 해상도 선택 ───────────────────────────────
# D435 지원 해상도
#   Color : 1920×1080 / 1280×720 / 848×480 / 640×480 / 424×240  @ 30fps
#   Depth : 1280×720  / 848×480  / 640×480 / 480×270 / 424×240  @ 30fps
#
# rs.align(color) 동작 원리:
#   align 후 depth frame은 color frame의 해상도로 재투영됨.
#   → aligned depth의 해상도 = color 해상도
#   → YOLO bbox 좌표(color 기준) = aligned depth 좌표  (스케일 변환 불필요)
#
# 따라서 color와 depth의 원본 해상도가 달라도 align 후에는 좌표계가 자동으로 일치.
# 단, color 해상도가 높을수록 YOLO 추론 부하가 커지므로 848×480 권장.
#
# 선택지 A (권장): color=848×480,  depth=848×480   빠름, 부하 적음
# 선택지 B       : color=1280×720, depth=1280×720  해상도 높음, 연산 증가
# 선택지 C       : color=1920×1080, depth=848×480  고화질 color + 경량 depth
#                  (align 후 depth가 1920×1080으로 재투영되므로 좌표 일치)
COLOR_W, COLOR_H = 640, 480
DEPTH_W, DEPTH_H = 640, 480

model_name = "baseline_advanced_last_0"

# ─────────────────────────────────────────────
#  경로 설정
# ─────────────────────────────────────────────
weights_path = "/home/ubuntu/autonomous_driving/yolo/weights"
ncnn_path    = "/home/ubuntu/autonomous_driving/yolo/ncnn"
os.makedirs(ncnn_path, exist_ok=True)

model_path = os.path.join(ncnn_path, model_name + "_ncnn_model")
pt_path    = os.path.join(weights_path, model_name + ".pt")

print("PID:", os.getpid())
print(f"[INFO] Color 해상도: {COLOR_W}×{COLOR_H}")
print(f"[INFO] Depth 해상도: {DEPTH_W}×{DEPTH_H}  (align 후 → {COLOR_W}×{COLOR_H})")

# ─────────────────────────────────────────────
#  YOLO 모델 로드 / 변환
# ─────────────────────────────────────────────
if os.path.exists(model_path):
    print("[INFO] NCNN 모델 로드")
    model = YOLO(model_path, task="detect")
else:
    print("[INFO] NCNN 파일 없음 → 변환 시작")
    base_model = YOLO(pt_path)
    base_model.export(format="ncnn", imgsz=IMGSZ, half=False)

    generated_path = os.path.join(weights_path, model_name + "_ncnn_model")
    target_path    = os.path.join(ncnn_path, model_name + "_ncnn_model")
    shutil.move(generated_path, target_path)
    model = YOLO(target_path, task="detect")


# ─────────────────────────────────────────────
#  depth 값 추출 함수
#
#  [핵심]
#  align.process() 이후 aligned depth frame은 이미 color 해상도로 재투영됨.
#  따라서 YOLO bbox 좌표(color 기준)를 스케일 변환 없이 그대로 사용.
#  depth_frame을 별도로 resize하면 안 됨 (0값 오염 문제).
#  depth_raw는 매 프레임 루프에서 1회만 변환해서 인자로 넘김 (bbox 수만큼 중복 방지).
# ─────────────────────────────────────────────
def get_depth_in_box(depth_frame, depth_raw, x1, y1, x2, y2):
    """
    bounding box 내의 center / median depth를 동시에 반환합니다.

    depth_frame : aligned depth_frame (color 해상도로 재투영 완료)
    depth_raw   : np.asanyarray(depth_frame.get_data()) — 매 프레임 1회 변환
    x1,y1,x2,y2: color 해상도 기준 bbox 좌표 (align 후 depth와 좌표계 일치)

    반환값: (center_m, median_m)  단위: meter

      center — bbox 중심 픽셀 1개
               get_distance() 사용, depth scale 자동 적용
               빠르지만 중심이 invalid(0)이면 0.0 반환

      median — bbox 내 유효 픽셀(>0)의 중앙값
               이상치(튀는 값)에 강건 → 데이터 수집 시 권장
               D435 invalid 픽셀(반사·투명·범위 초과)은 raw값 0 → 제외
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # ── center ──────────────────────────────
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    center_m = depth_frame.get_distance(cx, cy)  # 자동으로 meter 단위 반환

    # ── median ──────────────────────────────
    roi   = depth_raw[y1:y2, x1:x2]     # shape: (dy, dx), dtype: uint16
    valid = roi[roi > 0]                 # 0(invalid) 제외

    if len(valid) == 0:
        median_m = 0.0
    else:
        # get_units(): uint16 raw → meter 변환 계수 (D435 기본: 0.001 = mm→m)
        scale    = depth_frame.get_units()
        median_m = float(np.median(valid)) * scale

    return center_m, median_m


# ─────────────────────────────────────────────
#  RealSense 파이프라인 설정
# ─────────────────────────────────────────────
pipeline = rs.pipeline()
config   = rs.config()

config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16,  30)

pipeline.start(config)

# depth → color 좌표계로 재투영 (D435 렌즈 오프셋 보정 + color 해상도로 정렬)
# align 후: aligned_depth.해상도 == color.해상도 == COLOR_W×COLOR_H
align = rs.align(rs.stream.color)

print("[INFO] RealSense 스트림 시작. 'q'를 누르면 종료합니다.")

try:
    while True:
        # ── 프레임 취득 ──────────────────────────────────────────────────
        frames  = pipeline.wait_for_frames()
        aligned = align.process(frames)
        #   aligned 내부:
        #     color frame  → COLOR_W×COLOR_H (원본 그대로)
        #     depth frame  → COLOR_W×COLOR_H (color 기준으로 재투영됨)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        #   이 시점에서 color와 depth는 동일한 해상도(COLOR_W×COLOR_H),
        #   동일한 좌표계를 가짐 → bbox 좌표 그대로 depth 조회 가능

        if not color_frame or not depth_frame:
            print("[WARN] 프레임 취득 실패, 재시도...")
            continue

        # color → numpy  (YOLO 입력: BGR uint8, COLOR_W×COLOR_H)
        color_image = np.asanyarray(color_frame.get_data())

        # depth → numpy  (매 프레임 1회 변환, bbox 수만큼 재사용)
        # shape: (COLOR_H, COLOR_W), dtype: uint16
        depth_raw = np.asanyarray(depth_frame.get_data())

        # ── YOLO 추론 ─────────────────────────────────────────────────────
        # COLOR_W×COLOR_H → letterbox 416×416 → 추론 → COLOR_W×COLOR_H 역변환
        # boxes.xyxy: COLOR_W×COLOR_H 기준 좌표 반환
        results = model(color_image, conf=CONF, imgsz=IMGSZ, verbose=False)
        boxes   = results[0].boxes

        # ── depth 추출 및 시각화 ──────────────────────────────────────────
        if boxes is not None and len(boxes) > 0:
            for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                x1, y1, x2, y2 = box.tolist()

                # 클램핑: YOLO가 간혹 경계 밖 좌표 반환 방지
                x1 = max(0, min(x1, COLOR_W - 1))
                y1 = max(0, min(y1, COLOR_H - 1))
                x2 = max(0, min(x2, COLOR_W - 1))
                y2 = max(0, min(y2, COLOR_H - 1))

                # aligned depth에서 bbox 좌표 그대로 depth 조회
                # (align이 이미 좌표계를 맞춰줬으므로 스케일 변환 불필요)
                center_m, median_m = get_depth_in_box(
                    depth_frame, depth_raw, x1, y1, x2, y2
                )

                print(
                    f"bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] | "
                    f"conf: {float(conf):.2f} | "
                    f"class: {int(cls)} | "
                    f"center: {center_m:.3f} m | "
                    f"median: {median_m:.3f} m"
                )

                # 시각화
                ix1, iy1 = int(x1), int(y1)
                ix2, iy2 = int(x2), int(y2)

                cv2.rectangle(color_image, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)

                label = (f"cls:{int(cls)} {float(conf):.2f} "
                         f"c:{center_m:.2f}m med:{median_m:.2f}m")
                cv2.putText(
                    color_image, label,
                    (ix1, max(iy1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        cv2.imshow("YOLO + RealSense Depth", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] 종료 완료")
