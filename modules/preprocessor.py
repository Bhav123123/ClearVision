"""
modules/preprocessor.py
------------------------
Frame preprocessing pipeline:
  Resize → BGR-to-RGB → Gaussian Blur → Histogram Equalization → Normalize
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Prepares raw BGR frames for face detection.

    Parameters
    ----------
    target_size  : tuple (width, height) — resize target, default (640, 480)
    blur_kernel  : tuple — Gaussian blur kernel size, default (3, 3)
    equalize     : bool  — apply CLAHE histogram equalization, default True
    normalize    : bool  — normalize pixel values to [0, 1], default False
                          (keep False for OpenCV/dlib which expect uint8)
    """

    def __init__(
        self,
        target_size: tuple = (640, 480),
        blur_kernel: tuple = (3, 3),
        equalize: bool = True,
        normalize: bool = False,
    ):
        self.target_size = target_size
        self.blur_kernel = blur_kernel
        self.equalize    = equalize
        self.normalize   = normalize

        # CLAHE for adaptive histogram equalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        logger.info(f"Preprocessor initialized — target_size={target_size}, "
                    f"blur={blur_kernel}, equalize={equalize}")

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline on a BGR frame.
        Returns an RGB uint8 frame ready for detection.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            logger.warning("Empty frame received in Preprocessor.")
            return frame_bgr

        try:
            frame = self._resize(frame_bgr)
            frame = self._bgr_to_rgb(frame)
            frame = self._gaussian_blur(frame)
            if self.equalize:
                frame = self._histogram_equalize(frame)
            if self.normalize:
                frame = self._normalize(frame)
            return frame
        except Exception as exc:
            logger.error(f"Preprocessing failed: {exc}", exc_info=True)
            return frame_bgr   # return original on error

    # ── Pipeline stages ───────────────────────────────────────────────────────
    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        tw, th = self.target_size
        if (w, h) != (tw, th):
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
        return frame

    def _bgr_to_rgb(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _gaussian_blur(self, frame: np.ndarray) -> np.ndarray:
        if self.blur_kernel and all(k > 0 for k in self.blur_kernel):
            frame = cv2.GaussianBlur(frame, self.blur_kernel, sigmaX=0)
        return frame

    def _histogram_equalize(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE on the Y (luminance) channel in YCrCb space."""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    @staticmethod
    def _normalize(frame: np.ndarray) -> np.ndarray:
        """Scale uint8 [0,255] → float32 [0.0, 1.0]."""
        return (frame.astype(np.float32) / 255.0)

    # ── Utility ───────────────────────────────────────────────────────────────
    def is_low_light(self, frame_bgr: np.ndarray, threshold: int = 60) -> bool:
        """Returns True if mean pixel value is below threshold (dark frame)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < threshold
