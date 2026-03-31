"""
modules/detector.py
-------------------
Face detection engine supporting three backends:
  - haar : Haar Cascade Classifier (OpenCV)
  - hog  : HOG + SVM (dlib via face_recognition)
  - cnn  : CNN ResNet (dlib via face_recognition)
"""

import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Try importing face_recognition (optional heavy dep)
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    logger.warning("face_recognition not installed — Haar Cascade only mode.")


class FaceDetector:
    """
    Unified face detection class.

    Parameters
    ----------
    model : str
        One of 'haar', 'hog', 'cnn'. Defaults to 'hog'.
    scale_factor : float
        Haar Cascade scaleFactor (e.g. 1.1).
    min_neighbors : int
        Haar Cascade minNeighbors (e.g. 5).
    min_size : tuple
        Haar Cascade minimum face size in pixels.
    """

    SUPPORTED_MODELS = ("haar", "hog", "cnn")

    def __init__(
        self,
        model: str = "hog",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple = (30, 30),
    ):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {self.SUPPORTED_MODELS}, got '{model}'")

        self.model        = model
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size     = min_size

        # Load Haar cascade (always available as fallback)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        if model in ("hog", "cnn") and not FACE_REC_AVAILABLE:
            logger.warning(f"Model '{model}' requested but face_recognition is not installed. "
                           "Falling back to Haar Cascade.")
            self.model = "haar"

        logger.info(f"FaceDetector initialized with model='{self.model}'")

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, frame_rgb: np.ndarray, max_faces: int = 20) -> list[dict]:
        """
        Detect faces in an RGB frame.

        Returns
        -------
        list of dict, each containing:
            top, right, bottom, left : int  — bounding box coords
            confidence               : float — detection score (0–1)
            cx, cy                   : int  — face center pixel
            width, height            : int  — bounding box size
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return []

        try:
            if self.model == "haar":
                faces = self._detect_haar(frame_rgb)
            elif self.model == "hog":
                faces = self._detect_hog(frame_rgb)
            elif self.model == "cnn":
                faces = self._detect_cnn(frame_rgb)
            else:
                faces = []
        except Exception as exc:
            logger.error(f"Detection error with model '{self.model}': {exc}", exc_info=True)
            faces = []

        # Limit to max_faces (largest faces first)
        faces.sort(key=lambda f: (f["width"] * f["height"]), reverse=True)
        return faces[:max_faces]

    def switch_model(self, model: str):
        """Hot-swap the detection backend at runtime."""
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model}")
        if model in ("hog", "cnn") and not FACE_REC_AVAILABLE:
            raise RuntimeError("face_recognition is required for hog/cnn models.")
        self.model = model
        logger.info(f"Switched to model: {model}")

    # ── Private detection backends ────────────────────────────────────────────
    def _detect_haar(self, frame_rgb: np.ndarray) -> list[dict]:
        """Haar Cascade detection on grayscale frame."""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        results = []
        if len(detections) == 0:
            return results
        for (x, y, w, h) in detections:
            results.append(self._build_face(
                top=y, right=x + w, bottom=y + h, left=x, confidence=0.85
            ))
        return results

    def _detect_hog(self, frame_rgb: np.ndarray) -> list[dict]:
        """HOG + SVM detection using face_recognition library."""
        locations = face_recognition.face_locations(frame_rgb, model="hog")
        return [
            self._build_face(top=t, right=r, bottom=b, left=l, confidence=0.92)
            for (t, r, b, l) in locations
        ]

    def _detect_cnn(self, frame_rgb: np.ndarray) -> list[dict]:
        """CNN (ResNet-based dlib) detection via face_recognition library."""
        locations = face_recognition.face_locations(frame_rgb, model="cnn")
        return [
            self._build_face(top=t, right=r, bottom=b, left=l, confidence=0.97)
            for (t, r, b, l) in locations
        ]

    # ── Helper ────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_face(top, right, bottom, left, confidence) -> dict:
        w = right - left
        h = bottom - top
        return {
            "top":        top,
            "right":      right,
            "bottom":     bottom,
            "left":       left,
            "width":      w,
            "height":     h,
            "cx":         left + w // 2,
            "cy":         top  + h // 2,
            "confidence": round(confidence, 3),
        }
