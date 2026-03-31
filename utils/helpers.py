"""
utils/helpers.py
-----------------
Shared utility functions used across the Smart Vision pipeline.
"""

import cv2
import time
import numpy as np
from datetime import datetime


def get_frame_fps(prev_time: float) -> float:
    """Calculate frames per second given previous frame timestamp."""
    now = time.time()
    elapsed = now - prev_time
    return 1.0 / elapsed if elapsed > 0 else 0.0


def overlay_stats(frame_bgr: np.ndarray, face_count: int) -> np.ndarray:
    """
    Draw a semi-transparent stats bar at the top of the frame.
    Shows: face count + timestamp.
    """
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # Dark top banner
    cv2.rectangle(overlay, (0, 0), (w, 32), (13, 27, 42), -1)
    cv2.addWeighted(overlay, 0.7, frame_bgr, 0.3, 0, frame_bgr)

    ts    = datetime.now().strftime("%H:%M:%S")
    label = f"Smart Vision  |  Faces: {face_count}  |  {ts}"
    cv2.putText(frame_bgr, label, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 230, 255), 1, cv2.LINE_AA)
    return frame_bgr


def resize_keep_aspect(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Resize frame to max_width while preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale  = max_width / w
    new_h  = int(h * scale)
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def is_valid_frame(frame) -> bool:
    """Return True if frame is a non-None, non-empty numpy array."""
    return frame is not None and isinstance(frame, np.ndarray) and frame.size > 0


def apply_nms(faces: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    """
    Non-Maximum Suppression: remove overlapping bounding boxes.
    Keeps the box with the highest confidence when IoU > threshold.
    """
    if len(faces) <= 1:
        return faces

    boxes = np.array([[f["left"], f["top"], f["right"], f["bottom"]] for f in faces])
    scores = np.array([f["confidence"] for f in faces])
    indices = cv2.dnn.NMSBoxes(
        bboxes   = boxes.tolist(),
        scores   = scores.tolist(),
        score_threshold = 0.3,
        nms_threshold   = iou_threshold,
    )
    if len(indices) == 0:
        return []
    return [faces[i] for i in indices.flatten()]


def draw_grid(frame: np.ndarray, rows: int = 3, cols: int = 3,
              color=(50, 50, 50), alpha: float = 0.3) -> np.ndarray:
    """Overlay a faint grid on the frame (useful for debug mode)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for i in range(1, rows):
        y = int(h * i / rows)
        cv2.line(overlay, (0, y), (w, y), color, 1)
    for j in range(1, cols):
        x = int(w * j / cols)
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def format_duration(seconds: int) -> str:
    """Format integer seconds to 'Xh Ym Zs' string."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    parts = []
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
