import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load YOLOv11 TensorRT or PyTorch model
#model = YOLO("yolo11m.engine")                  # COCO model
model = YOLO("yolo-phone-detector.engine")       # Custom model

# store track history
track_history = {}

# thresholds
MOTION_THRESH = 15       # pixels displacement threshold
STATIC_FRAMES = 30       # how many frames to check for "static"
CONF_THRESHOLD = 0.4     # confidence threshold for phone detection

def is_static(track_id, bbox, frame_height):
    """
    Check if tracked object has been static for STATIC_FRAMES frames,
    but only discard if it's in lower region (table/lap/seat).
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if track_id not in track_history:
        track_history[track_id] = []
    track_history[track_id].append((cx, cy))

    if len(track_history[track_id]) > STATIC_FRAMES:
        track_history[track_id].pop(0)

    if len(track_history[track_id]) == STATIC_FRAMES:
        pts = np.array(track_history[track_id])
        dist = np.linalg.norm(pts[-1] - pts[0])

        # Discard if phone in lower half and not moving
        if dist < MOTION_THRESH and cy > frame_height // 2:
            return True  # static phone
    return False


def process_video(video_path, model, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return None

    # video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{os.path.basename(video_path)} | Resolution: {width}x{height} | FPS: {fps:.2f}")

    # output path
    out_path = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    detections_summary = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # results = model.track(frame, tracker="bytetrack.yaml",
        #                       classes=[67], imgsz=1280, verbose=False, conf=CONF_THRESHOLD)[0]

        results = model.track(frame, tracker="bytetrack.yaml", imgsz=640, verbose=False, conf=CONF_THRESHOLD)[0]      

        if results.boxes is not None:
            for det in results.boxes:
                track_id = int(det.id.item()) if det.id is not None else -1
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = float(det.conf.item())

                # static check
                if track_id != -1 and is_static(track_id, (x1, y1, x2, y2), frame.shape[0]):
                    continue

                # draw bbox + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                cv2.putText(frame,
                            f"{conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0), 2)

                detections_summary.append({
                    "video": os.path.basename(video_path),
                    "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "track_id": track_id,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        out.write(frame)

    cap.release()
    out.release()

    return pd.DataFrame(detections_summary)


if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []

    for file in os.listdir(input_dir):
        if file.lower().endswith((".mp4", ".avi", ".mov")):
            df = process_video(os.path.join(input_dir, file), model, output_dir)
            if df is not None:
                all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv("detections_summary.csv", index=False)
        print("Summary report saved")
