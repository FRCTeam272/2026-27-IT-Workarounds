#!/usr/bin/env python3
"""Detect persons in a webcam feed or image/video file using OpenCV DNN + YOLOv5n (ONNX)."""

import argparse
import ctypes
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILE = MODEL_DIR / "yolov5n.onnx"
MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"

INPUT_SIZE = (640, 640)
PERSON_CLASS = 0   # COCO class 0 = person
DEFAULT_CONF = 0.4
DEFAULT_NMS = 0.45
DEFAULT_SOUND = Path(__file__).parent / "sound.mp3"
DEFAULT_COOLDOWN = 10.0
DEFAULT_WAIT_TIME = 5.0


def toggle_media_playback():
    """Toggle system media play/pause state (Windows VK_MEDIA_PLAY_PAUSE)."""
    if sys.platform == "win32":
        try:
            VK_MEDIA_PLAY_PAUSE = 0xB3
            KEYEVENTF_KEYUP = 0x0002
            ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
            ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_KEYUP, 0)
        except Exception as exc:
            print(f"\n[Alert] Error toggling media: {exc}")


def play_sound(sound_path: Path):
    """Play a sound file (MP3/WAV) using Windows MCI or system fallback."""
    path_obj = Path(sound_path).resolve()
    if not path_obj.exists():
        print(f"\n[Alert] Warning: Sound file not found: '{path_obj}'")
        if sys.platform == "win32":
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                pass
        return

    if sys.platform == "win32":
        try:
            mci = ctypes.windll.winmm.mciSendStringW
            buf = ctypes.create_unicode_buffer(500)
            res = ctypes.windll.kernel32.GetShortPathNameW(str(path_obj), buf, 500)
            target = buf.value if res > 0 else str(path_obj)
            alias = "door_detector_alert"
            mci(f"close {alias}", None, 0, 0)
            err = mci(f'open "{target}" type mpegvideo alias {alias}', None, 0, 0)
            if err != 0:
                err = mci(f'open "{target}" alias {alias}', None, 0, 0)
            if err == 0:
                mci(f"play {alias} wait", None, 0, 0)
                mci(f"close {alias}", None, 0, 0)
                return
        except Exception as exc:
            print(f"\n[Alert] Error playing sound via MCI: {exc}")

    if sys.platform == "win32":
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            pass


class AlertController:
    """Manages cooldown and executes the media pause -> sound -> wait -> resume alert flow."""

    def __init__(self, sound_path: Path, cooldown: float = DEFAULT_COOLDOWN, wait_time: float = DEFAULT_WAIT_TIME):
        self.sound_path = Path(sound_path)
        self.cooldown = cooldown
        self.wait_time = wait_time
        self.last_triggered_time = 0.0
        self.is_running = False
        self._lock = threading.Lock()

    def trigger(self):
        with self._lock:
            now = time.time()
            if self.is_running or (now - self.last_triggered_time) < self.cooldown:
                return False
            self.is_running = True

        thread = threading.Thread(target=self._run_alert_sequence, daemon=True)
        thread.start()
        return True

    def _run_alert_sequence(self):
        try:
            print("\n[Alert] Person detected! Pausing media playback...")
            toggle_media_playback()
            print(f"[Alert] Playing sound '{self.sound_path.name}'...")
            play_sound(self.sound_path)
            print(f"[Alert] Waiting {self.wait_time:.1f} seconds...")
            time.sleep(self.wait_time)
            print("[Alert] Resuming media playback...")
            toggle_media_playback()
        finally:
            with self._lock:
                self.last_triggered_time = time.time()
                self.is_running = False


def _progress(blocks, block_size, total):
    pct = min(blocks * block_size / total * 100, 100)
    print(f"\r  {pct:.0f}%", end="", flush=True)


def download_model():
    MODEL_DIR.mkdir(exist_ok=True)
    if MODEL_FILE.exists():
        return
    print(f"Downloading {MODEL_FILE.name} (~3.8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE, reporthook=_progress)
        print()
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print(f"Download manually from:\n  {MODEL_URL}\nand place at:\n  {MODEL_FILE}")
        sys.exit(1)


def load_net():
    download_model()
    return cv2.dnn.readNetFromONNX(str(MODEL_FILE))


def detect(net, frame, conf_thresh, nms_thresh):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, INPUT_SIZE, swapRB=True, crop=False)
    net.setInput(blob)
    raw = net.forward()[0]  # (25200, 85): cx cy bw bh obj cls0..cls79

    boxes, scores = [], []
    for det in raw:
        objectness = float(det[4])
        class_id = int(np.argmax(det[5:]))
        confidence = objectness * float(det[5 + class_id])
        if class_id != PERSON_CLASS or confidence < conf_thresh:
            continue
        cx, cy, bw, bh = det[:4]
        x1 = int((cx - bw / 2) * w / INPUT_SIZE[0])
        y1 = int((cy - bh / 2) * h / INPUT_SIZE[1])
        boxes.append([x1, y1, int(bw * w / INPUT_SIZE[0]), int(bh * h / INPUT_SIZE[1])])
        scores.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, nms_thresh)
    if len(indices) == 0:
        return [], []
    idx = np.array(indices).flatten()
    return [boxes[i] for i in idx], [scores[i] for i in idx]


def annotate(frame, boxes, scores):
    for (x, y, bw, bh), score in zip(boxes, scores):
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.0%}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"Persons: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def run_webcam(net, conf, nms, alert_controller):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Error: could not open webcam.")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, scores = detect(net, frame, conf, nms)
        if boxes:
            alert_controller.trigger()
        print(f"\rPerson in frame: {bool(boxes)} ({len(boxes)} detected)", end="", flush=True)
        annotate(frame, boxes, scores)
        cv2.imshow("Person Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print()
    cap.release()
    cv2.destroyAllWindows()


def run_image(net, path, conf, nms, alert_controller):
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Error: could not read '{path}'.")
    boxes, scores = detect(net, frame, conf, nms)
    if boxes:
        alert_controller.trigger()
    print(f"Person in frame: {bool(boxes)} ({len(boxes)} detected)")
    annotate(frame, boxes, scores)
    cv2.imshow("Person Detector", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(net, path, conf, nms, alert_controller):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Error: could not open '{path}'.")
    print("Press 'q' to quit.")
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        boxes, scores = detect(net, frame, conf, nms)
        if boxes:
            alert_controller.trigger()
        print(f"\rFrame {frame_num} — Person: {bool(boxes)} ({len(boxes)})", end="", flush=True)
        annotate(frame, boxes, scores)
        cv2.imshow("Person Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Detect persons in a frame.")
    parser.add_argument("source", nargs="?", default="webcam",
                        help="Image/video path, or 'webcam' (default).")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONF,
                        help=f"Confidence threshold (default: {DEFAULT_CONF}).")
    parser.add_argument("--nms", type=float, default=DEFAULT_NMS,
                        help=f"NMS IoU threshold (default: {DEFAULT_NMS}).")
    parser.add_argument("--sound", type=Path, default=DEFAULT_SOUND,
                        help=f"Path to alert sound MP3 file (default: {DEFAULT_SOUND.name}).")
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                        help=f"Cooldown in seconds after alert finishes (default: {DEFAULT_COOLDOWN}s).")
    parser.add_argument("--wait-time", type=float, default=DEFAULT_WAIT_TIME,
                        help=f"Seconds to wait after sound before resuming media (default: {DEFAULT_WAIT_TIME}s).")
    args = parser.parse_args()

    net = load_net()
    alert_controller = AlertController(sound_path=args.sound, cooldown=args.cooldown, wait_time=args.wait_time)

    if args.source == "webcam":
        run_webcam(net, args.confidence, args.nms, alert_controller)
    elif args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        run_image(net, args.source, args.confidence, args.nms, alert_controller)
    else:
        run_video(net, args.source, args.confidence, args.nms, alert_controller)


if __name__ == "__main__":
    main()
