from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video import load_json, save_json


@dataclass
class BaselineResult:
    score: float
    yellow_ratio: float
    pred: str
    heatmap: np.ndarray


class YellowCapBaseline:
    """Simple cap-presence baseline based on yellow/orange pixels in the ROI.

    The anomaly score increases when the yellow cap-like area is lower than
    the normal training median. It is deliberately transparent and fast, so it
    works as an engineering baseline against AnomalyDINO.
    """

    def __init__(self, hsv_lower=(12, 35, 60), hsv_upper=(48, 255, 255), min_area_px: int = 25):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area_px = int(min_area_px)
        self.normal_ratio_median = 0.0
        self.normal_ratio_iqr = 0.0
        self.threshold = 0.08

    def _yellow_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if self.min_area_px > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            keep = np.zeros_like(mask)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= self.min_area_px:
                    keep[labels == i] = 255
            mask = keep
        return mask

    def yellow_ratio(self, roi_bgr: np.ndarray) -> float:
        mask = self._yellow_mask(roi_bgr)
        return float(np.count_nonzero(mask) / mask.size)

    def score_from_ratio(self, ratio: float) -> float:
        denom = max(self.normal_ratio_median, 1e-6)
        return float(max(0.0, (self.normal_ratio_median - ratio) / denom))

    def fit(self, rois_bgr: list[np.ndarray], threshold_policy: dict[str, Any]) -> dict[str, Any]:
        ratios = np.array([self.yellow_ratio(roi) for roi in rois_bgr], dtype=np.float32)
        if ratios.size == 0:
            raise RuntimeError("No training ROIs available for the color baseline.")
        self.normal_ratio_median = float(np.median(ratios))
        q1, q3 = np.percentile(ratios, [25, 75])
        self.normal_ratio_iqr = float(q3 - q1)
        scores = np.array([self.score_from_ratio(float(r)) for r in ratios], dtype=np.float32)
        percentile = float(threshold_policy.get("percentile", 99.5))
        std_factor = float(threshold_policy.get("std_factor", 3.0))
        min_threshold = float(threshold_policy.get("min_threshold", 0.08))
        threshold_p = float(np.percentile(scores, percentile))
        threshold_std = float(np.mean(scores) + std_factor * np.std(scores))
        self.threshold = max(threshold_p, threshold_std, min_threshold)
        return {
            "normal_ratio_median": self.normal_ratio_median,
            "normal_ratio_iqr": self.normal_ratio_iqr,
            "threshold": self.threshold,
            "train_ratio_min": float(np.min(ratios)),
            "train_ratio_max": float(np.max(ratios)),
            "train_score_mean": float(np.mean(scores)),
            "train_score_std": float(np.std(scores)),
            "train_count": int(ratios.size),
        }

    def predict(self, roi_bgr: np.ndarray) -> BaselineResult:
        ratio = self.yellow_ratio(roi_bgr)
        score = self.score_from_ratio(ratio)
        mask = self._yellow_mask(roi_bgr)
        inv = 255 - mask
        heatmap = cv2.GaussianBlur(inv, (0, 0), 7)
        pred = "NG" if score > self.threshold else "OK"
        return BaselineResult(score=score, yellow_ratio=ratio, pred=pred, heatmap=heatmap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hsv_lower": self.hsv_lower.tolist(),
            "hsv_upper": self.hsv_upper.tolist(),
            "min_area_px": self.min_area_px,
            "normal_ratio_median": self.normal_ratio_median,
            "normal_ratio_iqr": self.normal_ratio_iqr,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "YellowCapBaseline":
        obj = cls(data["hsv_lower"], data["hsv_upper"], data.get("min_area_px", 25))
        obj.normal_ratio_median = float(data["normal_ratio_median"])
        obj.normal_ratio_iqr = float(data.get("normal_ratio_iqr", 0.0))
        obj.threshold = float(data["threshold"])
        return obj

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        payload = self.to_dict()
        if extra:
            payload.update(extra)
        save_json(payload, path)

    @classmethod
    def load(cls, path: Path) -> "YellowCapBaseline":
        return cls.from_dict(load_json(path))
