"""
modules/reporter.py
--------------------
Loads attendance CSV data and generates:
  - Bar chart   : detections by hour of day
  - Pie chart   : detected vs. non-detected frames (session estimate)
  - Summary CSV : cleaned, de-duplicated attendance report
  - PNG charts  : saved to report_dir for embedding in dashboard
"""

import os
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates visual reports from attendance log data.

    Parameters
    ----------
    log_path   : str — path to attendance.csv
    report_dir : str — directory to save generated report files
    """

    def __init__(self, log_path: str = "logs/attendance.csv", report_dir: str = "reports"):
        self.log_path   = log_path
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────
    def generate(self) -> str | None:
        """
        Full report generation pipeline.
        Returns path to the summary CSV, or None if no data available.
        """
        df = self._load_data()
        if df is None or df.empty:
            logger.warning("No attendance data found — report not generated.")
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.bar_chart_path  = os.path.join(self.report_dir, f"detections_by_hour_{ts}.png")
        self.pie_chart_path  = os.path.join(self.report_dir, f"detection_ratio_{ts}.png")
        self.summary_csv     = os.path.join(self.report_dir, f"attendance_report_{ts}.csv")

        self._bar_chart_by_hour(df)
        self._pie_chart_ratio(df)
        self._export_summary_csv(df)

        logger.info(f"Report generated: {self.summary_csv}")
        return self.summary_csv

    def get_bar_chart_path(self) -> str | None:
        return getattr(self, "bar_chart_path", None)

    def get_pie_chart_path(self) -> str | None:
        return getattr(self, "pie_chart_path", None)

    # ── Private helpers ───────────────────────────────────────────────────────
    def _load_data(self) -> pd.DataFrame | None:
        if not os.path.exists(self.log_path):
            return None
        try:
            df = pd.read_csv(self.log_path)
            df.dropna(inplace=True)
            df.drop_duplicates(subset=["ID", "Date", "Time"], inplace=True)
            df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                            format="%Y-%m-%d %H:%M:%S", errors="coerce")
            df["Hour"] = df["DateTime"].dt.hour
            return df
        except Exception as exc:
            logger.error(f"Failed to load attendance data: {exc}", exc_info=True)
            return None

    def _bar_chart_by_hour(self, df: pd.DataFrame):
        hourly = df.groupby("Hour").size().reindex(range(24), fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(hourly.index, hourly.values,
                      color="#1E88E5", edgecolor="#0D47A1", linewidth=0.5)
        ax.set_xlabel("Hour of Day", fontsize=11)
        ax.set_ylabel("Detections", fontsize=11)
        ax.set_title("Face Detections by Hour of Day", fontsize=13, fontweight="bold")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate peak bar
        peak_hour = int(hourly.idxmax())
        ax.bar(peak_hour, hourly[peak_hour], color="#FF6D00", label=f"Peak: {peak_hour:02d}:00")
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(self.bar_chart_path, dpi=120)
        plt.close(fig)
        logger.info(f"Bar chart saved: {self.bar_chart_path}")

    def _pie_chart_ratio(self, df: pd.DataFrame):
        total_detections = len(df)
        unique_faces     = df["ID"].nunique()
        repeat_events    = total_detections - unique_faces

        labels = ["First-Time Detections", "Repeat Detections"]
        sizes  = [unique_faces, max(repeat_events, 0)]
        colors = ["#00897B", "#1E88E5"]
        explode= (0.05, 0)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sizes, labels=labels, colors=colors, explode=explode,
               autopct="%1.1f%%", startangle=140,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title("Detection Breakdown", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(self.pie_chart_path, dpi=120)
        plt.close(fig)
        logger.info(f"Pie chart saved: {self.pie_chart_path}")

    def _export_summary_csv(self, df: pd.DataFrame):
        summary_cols = ["ID", "Date", "Time", "Confidence"]
        df[summary_cols].to_csv(self.summary_csv, index=False)
        logger.info(f"Summary CSV exported: {self.summary_csv}")
