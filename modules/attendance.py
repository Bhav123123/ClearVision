"""
modules/attendance.py
----------------------
Logs face detection events to a CSV file using Pandas.
Implements per-face de-duplication within a configurable time window
to avoid thousands of redundant entries from a single sustained detection.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

COLUMNS = ["ID", "Date", "Time", "Confidence", "Timestamp"]


class AttendanceLogger:
    """
    Thread-safe (single-process) attendance CSV logger.

    Parameters
    ----------
    log_path       : str   — path to attendance CSV file
    dedup_interval : int   — seconds between repeated logs of the same face ID
    """

    def __init__(self, log_path: str = "logs/attendance.csv", dedup_interval: int = 30):
        self.log_path       = log_path
        self.dedup_interval = dedup_interval
        self._cache: dict[str, datetime] = {}   # face_id → last logged time
        self._records: list[dict]         = []   # in-memory records this session

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Load existing data if file exists
        if os.path.exists(log_path):
            try:
                existing = pd.read_csv(log_path)
                self._records = existing.to_dict("records")
                logger.info(f"Loaded {len(self._records)} existing attendance records.")
            except Exception as exc:
                logger.warning(f"Could not load existing attendance CSV: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────
    def log(self, face_id: str, confidence: float = 1.0) -> bool:
        """
        Log a detection event for face_id.

        Returns True if the event was newly logged, False if de-duplicated.
        """
        now = datetime.now()

        # De-duplication check
        if face_id in self._cache:
            elapsed = (now - self._cache[face_id]).total_seconds()
            if elapsed < self.dedup_interval:
                return False   # duplicate — skip

        # Record the event
        record = {
            "ID":          face_id,
            "Date":        now.strftime("%Y-%m-%d"),
            "Time":        now.strftime("%H:%M:%S"),
            "Confidence":  round(confidence, 3),
            "Timestamp":   now.isoformat(),
        }
        self._records.append(record)
        self._cache[face_id] = now

        # Persist to CSV
        self._write_csv(record)
        logger.info(f"Logged: {record}")
        return True

    def get_dataframe(self) -> pd.DataFrame:
        """Return all session records as a Pandas DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=COLUMNS)
        return pd.DataFrame(self._records)[COLUMNS]

    def clear_session(self):
        """Clear in-memory records and de-dup cache (does not delete CSV)."""
        self._records.clear()
        self._cache.clear()
        logger.info("Attendance session cleared.")

    def get_summary(self) -> dict:
        """Return aggregate stats for the current session."""
        df = self.get_dataframe()
        if df.empty:
            return {"total": 0, "unique_faces": 0, "peak_hour": "N/A"}
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
        return {
            "total":        len(df),
            "unique_faces": df["ID"].nunique(),
            "peak_hour":    f"{int(df['Hour'].mode()[0]):02d}:00",
        }

    # ── Private ───────────────────────────────────────────────────────────────
    def _write_csv(self, record: dict):
        """Append a single record to the CSV file."""
        try:
            df_row = pd.DataFrame([record])
            header = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
            df_row.to_csv(self.log_path, mode="a", header=header, index=False)
        except Exception as exc:
            logger.error(f"CSV write failed: {exc}", exc_info=True)
