"""
modules/annotator.py
---------------------
Draws bounding boxes, labels, confidence scores, timestamps,
and face counts onto frames. Also saves snapshot images.
"""

import cv2
import os
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class Annotator:
    """
    Renders detection results onto frames.

    Parameters
    ----------
    box_color       : BGR tuple for bounding box color
    label_color     : BGR tuple for label text
    score_color     : BGR tuple for confidence score text
    ts_color        : BGR tuple for timestamp text
    no_face_color   : BGR tuple for 'No Face Detected' text
    box_thickness   : line thickness for bounding box
    font_scale      : text size scale factor
    """

    def __init__(
        self,
        box_color:     tuple = (0, 255, 80),
        label_color:   tuple = (255, 255, 255),
        score_color:   tuple = (0, 255, 255),
        ts_color:      tuple = (200, 200, 0),
        no_face_color: tuple = (0, 0, 255),
        box_thickness: int   = 2,
        font_scale:    float = 0.55,
    ):
        self.box_color     = box_color
        self.label_color   = label_color
        self.score_color   = score_color
        self.ts_color      = ts_color
        self.no_face_color = no_face_color
        self.box_thickness = box_thickness
        self.font_scale    = font_scale
        self.font          = cv2.FONT_HERSHEY_SIMPLEX

    # ── Main draw method ──────────────────────────────────────────────────────
    def draw(self, frame_bgr: np.ndarray, faces: list[dict]) -> np.ndarray:
        """
        Draw all detected face annotations onto frame_bgr.
        Returns the annotated frame (modifies in-place copy).
        """
        if not faces:
            self._draw_no_face_label(frame_bgr)
        else:
            for idx, face in enumerate(faces):
                self._draw_box(frame_bgr, face)
                self._draw_label(frame_bgr, face, idx)
                self._draw_confidence(frame_bgr, face)

        self._draw_timestamp(frame_bgr)
        self._draw_face_count(frame_bgr, len(faces))
        return frame_bgr

    # ── Private drawing helpers ───────────────────────────────────────────────
    def _draw_box(self, frame: np.ndarray, face: dict):
        cv2.rectangle(
            frame,
            (face["left"], face["top"]),
            (face["right"], face["bottom"]),
            self.box_color,
            self.box_thickness,
        )

    def _draw_label(self, frame: np.ndarray, face: dict, idx: int):
        label = f"Face {idx + 1}"
        x = face["left"]
        y = max(face["top"] - 8, 14)
        # Background pill for readability
        (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
        cv2.rectangle(frame, (x - 1, y - th - 4), (x + tw + 2, y + 2),
                      self.box_color, -1)
        cv2.putText(frame, label, (x, y),
                    self.font, self.font_scale, self.label_color, 1, cv2.LINE_AA)

    def _draw_confidence(self, frame: np.ndarray, face: dict):
        conf_text = f"{face['confidence'] * 100:.0f}%"
        x = face["left"]
        y = face["bottom"] + 16
        cv2.putText(frame, conf_text, (x, y),
                    self.font, self.font_scale, self.score_color, 1, cv2.LINE_AA)

    def _draw_timestamp(self, frame: np.ndarray):
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        h  = frame.shape[0]
        cv2.putText(frame, ts, (8, h - 10),
                    self.font, 0.45, self.ts_color, 1, cv2.LINE_AA)

    def _draw_face_count(self, frame: np.ndarray, count: int):
        text = f"Faces: {count}"
        w    = frame.shape[1]
        (tw, _), _ = cv2.getTextSize(text, self.font, 0.55, 1)
        cv2.putText(frame, text, (w - tw - 10, 22),
                    self.font, 0.55, self.label_color, 1, cv2.LINE_AA)

    def _draw_no_face_label(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        text = "No Face Detected"
        (tw, th), _ = cv2.getTextSize(text, self.font, 0.9, 2)
        cv2.putText(frame, text,
                    ((w - tw) // 2, (h + th) // 2),
                    self.font, 0.9, self.no_face_color, 2, cv2.LINE_AA)

    # ── Snapshot saver ────────────────────────────────────────────────────────
    def save_snapshot(self, frame_bgr: np.ndarray, face: dict, snap_dir: str = "snapshots"):
        """Save a cropped face snapshot with timestamp filename."""
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            ts_str   = datetime.now().strftime("%H%M%S_%f")
            save_dir = os.path.join(snap_dir, date_str)
            os.makedirs(save_dir, exist_ok=True)

            # Crop with small padding
            pad = 10
            h, w = frame_bgr.shape[:2]
            top    = max(0, face["top"]    - pad)
            bottom = min(h, face["bottom"] + pad)
            left   = max(0, face["left"]   - pad)
            right  = min(w, face["right"]  + pad)

            cropped  = frame_bgr[top:bottom, left:right]
            filename = os.path.join(save_dir, f"face_{ts_str}.jpg")
            cv2.imwrite(filename, cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
            logger.debug(f"Snapshot saved: {filename}")
        except Exception as exc:
            logger.error(f"Snapshot save failed: {exc}", exc_info=True)
