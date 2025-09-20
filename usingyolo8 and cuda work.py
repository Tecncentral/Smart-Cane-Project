import argparse
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import pyttsx3
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

# ── Parameters ─────────────────────────────────────────────────────────────
ANNOUNCE_INTERVAL_S = 10
SKIP_FRAMES         = 1
INPUT_SIZE          = 416

DET_CONF_THRESH = 0.5
DET_IOU_THRESH  = 0.45

LABEL_COLOR    = (255, 0, 0)
TRACK_COLOR    = (0, 255, 0)
TEXT_THICKNESS = 1
FONT           = cv2.FONT_HERSHEY_SIMPLEX

tts_lock = threading.Lock()

class EMA:
    def __init__(self, box, alpha=0.6):
        self.box = np.array(box, float)
        self.alpha = alpha

    def update(self, new_box):
        nb = np.array(new_box, float)
        self.box = self.alpha * self.box + (1 - self.alpha) * nb
        return tuple(self.box.astype(int))

def direction_from_center(cx, w):
    r = cx / w
    if r < 0.2: return "left"
    elif r < 0.4: return "slightly left"
    elif r < 0.6: return "front"
    elif r < 0.8: return "slightly right"
    else: return "right"

def estimate_distance(box_h, h):
    rel = box_h / h
    if rel >= 0.60: return "very close"
    if rel >= 0.40: return "close"
    if rel >= 0.20: return "medium distance"
    return "far"

def speak(engine, text):
    with tts_lock:
        engine.say(text)
        engine.runAndWait()

def run(source, weights, save_tracks=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        wpath = Path(weights)
        if not wpath.exists():
            raise FileNotFoundError
        model = YOLO(str(wpath)).to(device)
    except Exception:
        print(f"[INFO] Auto-downloading model: {weights}")
        model = YOLO(weights).to(device)
        wpath = Path(weights)

    tracker = DeepSort(
        max_age=60,
        n_init=2,
        nms_max_overlap=0.5,
        max_iou_distance=0.7,
        embedder="mobilenet",
        embedder_gpu=torch.cuda.is_available(),
        half=torch.cuda.is_available()
    )

    tts = pyttsx3.init()
    tts.setProperty("rate", 175)

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"❌ Unable to open source {source}")

    announced = {}
    emas = {}
    mot_lines = []
    frame_idx = 0

    print("[INFO] Running. Press 'q' to quit…")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]
        detections = []

        if frame_idx % SKIP_FRAMES == 0:
            frame_resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            res = model(frame_resized,
                        imgsz=INPUT_SIZE,
                        device=device,
                        verbose=False,
                        conf=DET_CONF_THRESH,
                        iou=DET_IOU_THRESH)[0]

            for box in res.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = int(x1 * w0 / INPUT_SIZE)
                y1 = int(y1 * h0 / INPUT_SIZE)
                x2 = int(x2 * w0 / INPUT_SIZE)
                y2 = int(y2 * h0 / INPUT_SIZE)
                conf = float(box.conf[0])
                w_box, h_box = x2 - x1, y2 - y1
                if w_box < 5 or h_box < 5:
                    continue
                detections.append(([x1, y1, w_box, h_box], conf, cls))
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), LABEL_COLOR, 2)
                cv2.putText(frame, label, (x1, y1 - 10), FONT, 0.5, LABEL_COLOR, TEXT_THICKNESS)

        try:
            tracks = tracker.update_tracks(detections, frame=frame)
        except Exception as e:
            print(f"[WARN] tracker failed: {e}")
            tracks = []

        now = time.time()
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, trk.to_ltrb())
            tid = trk.track_id
            det_conf = getattr(trk, "det_conf", 1.0) or 0.0
            if save_tracks:
                mot_lines.append(f"{frame_idx+1},{tid},{x1},{y1},{x2-x1},{y2-y1},{det_conf:.2f},-1,-1,-1\n")
            if tid not in emas:
                emas[tid] = EMA((x1, y1, x2, y2))
            x1, y1, x2, y2 = emas[tid].update((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), TRACK_COLOR, 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10), FONT, 0.5, TRACK_COLOR, TEXT_THICKNESS)
            if trk.det_class == 0:
                cx = (x1 + x2) / 2
                dir_str = direction_from_center(cx, w0)
                dist_str = estimate_distance(y2 - y1, h0)
                phrase = f"There is a person {dist_str} {dir_str}."
                if tid not in announced or now - announced[tid] >= ANNOUNCE_INTERVAL_S:
                    print("[ANNOUNCE]", phrase)
                    threading.Thread(target=speak, args=(tts, phrase), daemon=True).start()
                    announced[tid] = now

        cv2.imshow("Smart Cane", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if save_tracks:
        Path(save_tracks).write_text("".join(mot_lines))
        print(f"[INFO] Saved tracks to {save_tracks}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="camera or video path")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model file name")
    p.add_argument("--save-tracks", help="Optional path to save MOT output")
    args = p.parse_args()
    run(args.source, args.model, args.save_tracks)
