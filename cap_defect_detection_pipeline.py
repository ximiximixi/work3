from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


Color = Tuple[int, int, int]


@dataclass
class CapDetectorConfig:
    right_half_x: int = 960
    crop_size: int = 96
    grid: int = 8
    max_memory_patches: int = 4096
    random_seed: int = 7
    sample_step_sec: float = 0.5
    train_step_sec: float = 2.0
    scan_start_sec: float = 0.0
    scan_end_sec: Optional[float] = None
    review_fps: float = 2.0
    review_step_sec: float = 0.5
    min_blue_area: int = 450
    pair_x_tolerance: float = 58.0
    pair_y_min: float = 420.0
    pair_y_max: float = 850.0
    min_event_observations: int = 2
    event_gap_sec: float = 1.25
    present_white_center_min: float = 0.45
    present_white_total_min: float = 0.56
    pseudo_normal_white_center_min: float = 0.62
    missing_white_center_max: float = 0.25
    missing_white_total_max: float = 0.36
    missing_blue_center_min: float = 0.18
    min_blur_var: float = 28.0


@dataclass
class BlueAnchor:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    confidence: float


@dataclass
class SleeveMetrics:
    white_center: float
    white_total: float
    blue_center: float
    blue_total: float
    blob_area_ratio: float
    blob_circularity: float
    blob_offset: float
    blur_var: float
    visibility: float
    clipped: bool


@dataclass
class EndResult:
    side: str
    state: str
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    white_center: float
    white_total: float
    blue_center: float
    blue_total: float
    blob_area_ratio: float
    blob_circularity: float
    blob_offset: float
    blur_var: float
    visibility: float
    patch_score: float
    anomaly_score: float
    reason: str


@dataclass
class CapObservation:
    time_sec: float
    frame_idx: int
    sample_index: int
    sample_cx: float
    sample_y1: int
    sample_y2: int
    paired: bool
    top_anchor_x: int
    top_anchor_y: int
    top_anchor_w: int
    top_anchor_h: int
    bottom_anchor_x: int
    bottom_anchor_y: int
    bottom_anchor_w: int
    bottom_anchor_h: int
    top_state: str
    bottom_state: str
    side_upper_state: str
    side_lower_state: str
    side_state: str
    raw_state: str
    final_state: str
    missing_side: str
    unknown_reason: str
    top_white_center: float
    bottom_white_center: float
    top_white_total: float
    bottom_white_total: float
    top_blue_center: float
    bottom_blue_center: float
    side_upper_white_center: float
    side_lower_white_center: float
    top_patch_score: float
    bottom_patch_score: float
    anomaly_score: float
    event_id: int


class CapPatchMemory:
    def __init__(self, cfg: CapDetectorConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.nn: Optional[NearestNeighbors] = None
        self.memory: Optional[np.ndarray] = None
        self.threshold: float = 1.0

    def _patch_features(self, crop_bgr: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        crop = cv2.resize(crop_bgr, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        cell = cfg.crop_size // cfg.grid
        feats = []
        for py in range(cfg.grid):
            for px in range(cfg.grid):
                y1 = py * cell
                x1 = px * cell
                y2 = cfg.crop_size if py == cfg.grid - 1 else (py + 1) * cell
                x2 = cfg.crop_size if px == cfg.grid - 1 else (px + 1) * cell
                hp = hsv[y1:y2, x1:x2]
                lp = lab[y1:y2, x1:x2]
                gp = gray[y1:y2, x1:x2]
                gradp = grad[y1:y2, x1:x2]
                lapp = lap[y1:y2, x1:x2]
                h = hp[:, :, 0] / 180.0
                s = hp[:, :, 1] / 255.0
                v = hp[:, :, 2] / 255.0
                white = ((hp[:, :, 1] < 85) & (hp[:, :, 2] > 105)).astype(np.float32)
                blue = ((hp[:, :, 0] >= 92) & (hp[:, :, 0] <= 132) & (hp[:, :, 1] > 55) & (hp[:, :, 2] > 35)).astype(np.float32)
                hue_angle = 2 * np.pi * h
                feats.append(
                    [
                        float(np.sin(hue_angle).mean()),
                        float(np.cos(hue_angle).mean()),
                        float(s.mean()),
                        float(s.std()),
                        float(v.mean()),
                        float(v.std()),
                        float((lp[:, :, 0] / 255.0).mean()),
                        float((lp[:, :, 1] / 255.0).mean()),
                        float((lp[:, :, 2] / 255.0).mean()),
                        float((gp / 255.0).mean()),
                        float((gp / 255.0).std()),
                        float((gradp / 255.0).mean()),
                        float((np.abs(lapp) / 255.0).mean()),
                        float(white.mean()),
                        float(blue.mean()),
                    ]
                )
        return np.asarray(feats, dtype=np.float32)

    def fit(self, crops: Sequence[np.ndarray]) -> List[float]:
        if not crops:
            raise RuntimeError("No pseudo-normal sleeve crops were collected for training.")
        patch_features = np.vstack([self._patch_features(crop) for crop in crops])
        scaled = self.scaler.fit_transform(patch_features)
        if scaled.shape[0] > self.cfg.max_memory_patches:
            kmeans = MiniBatchKMeans(
                n_clusters=self.cfg.max_memory_patches,
                random_state=self.cfg.random_seed,
                batch_size=4096,
                n_init="auto",
                reassignment_ratio=0.01,
            )
            kmeans.fit(scaled)
            self.memory = kmeans.cluster_centers_.astype(np.float32)
        else:
            self.memory = scaled.astype(np.float32)
        self.nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        self.nn.fit(self.memory)
        scores = [self.score(crop)[0] for crop in crops]
        arr = np.asarray(scores, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)) + 1e-6)
        self.threshold = max(float(np.quantile(arr, 0.975)), median + 6.0 * mad, 0.75)
        return scores

    def score(self, crop_bgr: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        if self.nn is None:
            raise RuntimeError("Patch memory is not fitted.")
        feats = self._patch_features(crop_bgr)
        scaled = self.scaler.transform(feats)
        distances, _ = self.nn.kneighbors(scaled, return_distance=True)
        patch_scores = distances[:, 0].astype(np.float32)
        top_k = max(1, int(math.ceil(len(patch_scores) * 0.15)))
        image_score = float(np.sort(patch_scores)[-top_k:].mean())
        heat = patch_scores.reshape(self.cfg.grid, self.cfg.grid)
        return image_score, heat, patch_scores


def pil_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def open_video(video_path: Path) -> Tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, frame_count, width, height


def read_frame_at(cap: cv2.VideoCapture, time_sec: float) -> Tuple[bool, Optional[np.ndarray], int]:
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ok, frame = cap.read()
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    return ok, frame if ok else None, frame_idx


def iter_times(start: float, end: float, step: float) -> Iterable[float]:
    t = start
    while t <= end + 1e-6:
        yield round(t, 4)
        t += step


def detect_blue_anchors(frame_bgr: np.ndarray, cfg: CapDetectorConfig) -> List[BlueAnchor]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (92, 55, 35), (132, 255, 255))
    mask[:, : cfg.right_half_x] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    anchors: List[BlueAnchor] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = cents[i]
        aspect = w / max(1, h)
        if x < cfg.right_half_x:
            continue
        if not (area >= cfg.min_blue_area and 35 <= w <= 190 and 8 <= h <= 90 and aspect >= 1.08):
            continue
        confidence = min(1.0, area / max(1.0, w * h)) * min(1.0, aspect / 2.0)
        anchors.append(BlueAnchor(int(x), int(y), int(w), int(h), int(area), float(cx), float(cy), float(confidence)))
    anchors.sort(key=lambda c: (c.cx, c.cy))
    return anchors


def pair_anchors(anchors: Sequence[BlueAnchor], cfg: CapDetectorConfig) -> Tuple[List[Tuple[BlueAnchor, BlueAnchor]], List[BlueAnchor]]:
    pairs: List[Tuple[BlueAnchor, BlueAnchor]] = []
    used: set[int] = set()
    for i, top in enumerate(anchors):
        if i in used:
            continue
        best_idx = None
        best_cost = float("inf")
        for j, bottom in enumerate(anchors):
            if i == j or j in used or bottom.cy <= top.cy:
                continue
            dx = abs(bottom.cx - top.cx)
            dy = bottom.cy - top.cy
            if dx > cfg.pair_x_tolerance or not (cfg.pair_y_min <= dy <= cfg.pair_y_max):
                continue
            cost = dx + 0.012 * abs(dy - 650.0) + 0.003 * abs(bottom.w - top.w)
            if cost < best_cost:
                best_idx = j
                best_cost = cost
        if best_idx is not None:
            used.add(i)
            used.add(best_idx)
            pairs.append((top, anchors[best_idx]))
    unpaired = [a for idx, a in enumerate(anchors) if idx not in used]
    pairs.sort(key=lambda p: p[0].cx)
    return pairs, unpaired


def sleeve_roi(frame_shape: Tuple[int, int, int], anchor: BlueAnchor, side: str, cfg: CapDetectorConfig) -> Tuple[int, int, int, int, bool]:
    height, width = frame_shape[:2]
    roi_w = int(max(30, min(58, anchor.w * 0.28)))
    roi_h = int(max(34, min(52, anchor.h * 0.58)))
    cx = anchor.cx
    if side == "top":
        y2 = int(anchor.y + anchor.h * 0.18)
        y1 = y2 - roi_h
    else:
        y1 = int(anchor.y + anchor.h * 0.82)
        y2 = y1 + roi_h
    x1 = int(round(cx - roi_w / 2))
    x2 = int(round(cx + roi_w / 2))
    clipped = x1 < cfg.right_half_x or y1 < 0 or x2 > width or y2 > height
    x1 = max(cfg.right_half_x, max(0, x1))
    x2 = min(width, x2)
    y1 = max(0, y1)
    y2 = min(height, y2)
    return x1, y1, x2, y2, clipped


def side_sleeve_rois(
    frame_shape: Tuple[int, int, int],
    top_anchor: BlueAnchor,
    bottom_anchor: BlueAnchor,
    cfg: CapDetectorConfig,
) -> List[Tuple[str, int, int, int, int, bool]]:
    candidates = side_sleeve_candidate_rois(frame_shape, top_anchor, bottom_anchor, cfg)
    return [(roi[0].removesuffix("_left"), *roi[1:]) for roi in candidates if roi[0].endswith("_left")]


def side_sleeve_candidate_rois(
    frame_shape: Tuple[int, int, int],
    top_anchor: BlueAnchor,
    bottom_anchor: BlueAnchor,
    cfg: CapDetectorConfig,
) -> List[Tuple[str, int, int, int, int, bool]]:
    height, width = frame_shape[:2]
    cap_w = max(top_anchor.w, bottom_anchor.w)
    cap_h = max(top_anchor.h, bottom_anchor.h)
    body_left = int(min(top_anchor.cx, bottom_anchor.cx) - cap_w * 0.43)
    body_right = int(max(top_anchor.cx, bottom_anchor.cx) + cap_w * 0.43)
    roi_w = int(max(70, min(125, cap_w * 0.82)))
    roi_h = int(max(58, min(110, cap_h * 1.20)))
    left_x2 = body_left + int(roi_w * 0.18)
    left_x1 = left_x2 - roi_w
    right_x1 = body_right - int(roi_w * 0.18)
    right_x2 = right_x1 + roi_w
    centers = {
        "side_upper": int(top_anchor.y + top_anchor.h + roi_h * 0.58),
        "side_lower": int(bottom_anchor.y - roi_h * 0.58),
    }
    rois = []
    for name, cy in centers.items():
        y1 = cy - roi_h // 2
        y2 = cy + roi_h // 2
        for suffix, x1, x2 in [("left", left_x1, left_x2), ("right", right_x1, right_x2)]:
            clipped = x1 < cfg.right_half_x or y1 < 0 or x2 > width or y2 > height
            rois.append((f"{name}_{suffix}", max(cfg.right_half_x, x1), max(0, y1), min(width, x2), min(height, y2), clipped))
    return rois


def sleeve_metrics(crop_bgr: np.ndarray, clipped: bool) -> SleeveMetrics:
    if crop_bgr.size == 0:
        return SleeveMetrics(0, 0, 0, 0, 0, 0, 1, 0, 0, True)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    white = ((s < 90) & (v > 105)).astype(np.uint8)
    blue = ((h >= 92) & (h <= 132) & (s > 55) & (v > 35)).astype(np.uint8)
    yy, xx = np.ogrid[: crop_bgr.shape[0], : crop_bgr.shape[1]]
    center = ((xx - crop_bgr.shape[1] / 2) ** 2 / max(1.0, (crop_bgr.shape[1] * 0.28) ** 2)) + (
        (yy - crop_bgr.shape[0] / 2) ** 2 / max(1.0, (crop_bgr.shape[0] * 0.43) ** 2)
    ) <= 1
    white_center = float(white[center].mean()) if np.any(center) else 0.0
    blue_center = float(blue[center].mean()) if np.any(center) else 0.0
    white_total = float(white.mean())
    blue_total = float(blue.mean())
    num, labels, stats, cents = cv2.connectedComponentsWithStats(white)
    best_area = 0
    best_circularity = 0.0
    best_offset = 1.0
    crop_center = np.asarray([crop_bgr.shape[1] / 2.0, crop_bgr.shape[0] / 2.0], dtype=np.float32)
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < best_area:
            continue
        x, y, w, h, _ = stats[idx]
        component = ((labels[y : y + h, x : x + w] == idx).astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(c, True) for c in cnts))
        circ = float(4.0 * math.pi * area / max(perimeter * perimeter, 1.0))
        offset = float(np.linalg.norm(np.asarray(cents[idx], dtype=np.float32) - crop_center) / max(crop_bgr.shape[:2]))
        best_area = area
        best_circularity = max(0.0, min(1.0, circ))
        best_offset = offset
    area_ratio = float(best_area / max(1, crop_bgr.shape[0] * crop_bgr.shape[1]))
    blur = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    visibility = min(1.0, crop_bgr.shape[0] * crop_bgr.shape[1] / max(1.0, 46.0 * 42.0))
    return SleeveMetrics(
        white_center=white_center,
        white_total=white_total,
        blue_center=blue_center,
        blue_total=blue_total,
        blob_area_ratio=area_ratio,
        blob_circularity=best_circularity,
        blob_offset=best_offset,
        blur_var=blur,
        visibility=visibility,
        clipped=clipped,
    )


def classify_end(
    side: str,
    frame_bgr: np.ndarray,
    anchor: Optional[BlueAnchor],
    cfg: CapDetectorConfig,
    memory: Optional[CapPatchMemory],
) -> Tuple[EndResult, np.ndarray, Optional[np.ndarray]]:
    if anchor is None:
        return (
            EndResult(side, "UNKNOWN", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2.0, "missing_anchor"),
            np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=np.uint8),
            None,
        )
    x1, y1, x2, y2, clipped = sleeve_roi(frame_bgr.shape, anchor, side, cfg)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return (
            EndResult(side, "UNKNOWN", x1, y1, x2, y2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2.0, "empty_roi"),
            np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=np.uint8),
            None,
        )
    crop_resized = cv2.resize(crop, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)
    m = sleeve_metrics(crop, clipped)
    patch_score = 0.0
    heat = None
    if memory is not None and memory.nn is not None:
        patch_score, heat, _ = memory.score(crop_resized)
    patch_norm = patch_score / max(memory.threshold if memory is not None else 1.0, 1e-6)
    evidence_present = (
        0.68 * m.white_center
        + 0.22 * m.white_total
        + 0.07 * min(1.0, m.blob_area_ratio / 0.42)
        + 0.03 * max(0.0, 1.0 - m.blob_offset)
    )
    missing_score = (
        max(0.0, cfg.present_white_center_min - m.white_center) * 2.2
        + max(0.0, cfg.present_white_total_min - m.white_total) * 1.1
        + max(0.0, m.blue_center - cfg.missing_blue_center_min) * 0.9
        + max(0.0, patch_norm - 1.0) * 0.38
    )
    state = "UNKNOWN"
    reason = "ambiguous_evidence"
    if m.visibility < 0.55:
        state, reason = "UNKNOWN", "roi_too_small_or_partial"
    elif m.blur_var < cfg.min_blur_var:
        state, reason = "UNKNOWN", "motion_blur_or_low_texture"
    elif patch_norm >= 1.30 and m.white_center < 0.58 and m.white_total < 0.66:
        state, reason = "MISSING", "patch_memory_outlier"
    elif m.blue_center >= 0.10 and patch_norm >= 1.05 and m.white_center < 0.52 and m.white_total < 0.60:
        state, reason = "MISSING", "exposed_blue_port"
    elif m.white_center < 0.44 and m.white_total < 0.56 and (
        m.blue_center >= 0.045 or m.blue_total >= 0.030 or patch_norm >= 1.00
    ):
        state, reason = "MISSING", "weak_beige_evidence_at_port"
    elif m.white_center >= cfg.present_white_center_min or m.white_total >= cfg.present_white_total_min:
        state, reason = "PRESENT", "beige_sleeve_visible"
    elif m.white_center <= cfg.missing_white_center_max and m.white_total <= cfg.missing_white_total_max and (
        m.blue_center >= 0.08 or m.blue_total >= 0.08 or patch_norm >= 1.20
    ):
        state, reason = "MISSING", "beige_sleeve_absent"
    result = EndResult(
        side=side,
        state=state,
        roi_x1=x1,
        roi_y1=y1,
        roi_x2=x2,
        roi_y2=y2,
        white_center=float(m.white_center),
        white_total=float(m.white_total),
        blue_center=float(m.blue_center),
        blue_total=float(m.blue_total),
        blob_area_ratio=float(m.blob_area_ratio),
        blob_circularity=float(m.blob_circularity),
        blob_offset=float(m.blob_offset),
        blur_var=float(m.blur_var),
        visibility=float(m.visibility),
        patch_score=float(patch_score),
        anomaly_score=float(missing_score),
        reason=reason,
    )
    return result, crop_resized, heat


def classify_side_roi(
    name: str,
    frame_bgr: np.ndarray,
    roi: Tuple[str, int, int, int, int, bool],
    cfg: CapDetectorConfig,
) -> Tuple[EndResult, np.ndarray]:
    _, x1, y1, x2, y2, clipped = roi
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return (
            EndResult(name, "UNKNOWN", x1, y1, x2, y2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1.8, "empty_side_roi"),
            np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=np.uint8),
        )
    resized = cv2.resize(crop, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)
    m = sleeve_metrics(crop, clipped)
    side_score = max(0.0, 0.58 - m.white_center) * 2.2 + max(0.0, 0.54 - m.white_total) * 0.9
    if m.visibility < 0.55:
        state, reason = "UNKNOWN", "side_roi_too_small"
    elif (
        m.white_center >= 0.72
        and (m.white_total >= 0.70 or m.blob_area_ratio >= 0.60)
        and m.blob_area_ratio >= 0.52
    ):
        state, reason = "PRESENT", "side_beige_sleeve_visible"
    elif m.white_center < 0.70 and m.white_total < 0.76:
        state, reason = "MISSING", "side_beige_sleeve_absent"
    elif m.white_center < 0.55 and m.white_total < 0.55 and m.blob_area_ratio < 0.43:
        state, reason = "MISSING", "side_beige_sleeve_absent"
    else:
        state, reason = "UNKNOWN", "side_weak_beige_evidence"
    return (
        EndResult(
            side=name,
            state=state,
            roi_x1=x1,
            roi_y1=y1,
            roi_x2=x2,
            roi_y2=y2,
            white_center=float(m.white_center),
            white_total=float(m.white_total),
            blue_center=float(m.blue_center),
            blue_total=float(m.blue_total),
            blob_area_ratio=float(m.blob_area_ratio),
            blob_circularity=float(m.blob_circularity),
            blob_offset=float(m.blob_offset),
            blur_var=float(m.blur_var),
            visibility=float(m.visibility),
            patch_score=0.0,
            anomaly_score=float(side_score),
            reason=reason,
        ),
        resized,
    )


def relabel_side_result(result: EndResult, side: str, reason: str) -> EndResult:
    return EndResult(
        side=side,
        state=result.state,
        roi_x1=result.roi_x1,
        roi_y1=result.roi_y1,
        roi_x2=result.roi_x2,
        roi_y2=result.roi_y2,
        white_center=result.white_center,
        white_total=result.white_total,
        blue_center=result.blue_center,
        blue_total=result.blue_total,
        blob_area_ratio=result.blob_area_ratio,
        blob_circularity=result.blob_circularity,
        blob_offset=result.blob_offset,
        blur_var=result.blur_var,
        visibility=result.visibility,
        patch_score=result.patch_score,
        anomaly_score=result.anomaly_score,
        reason=reason,
    )


def side_candidate_score(result: EndResult) -> float:
    base = (
        0.45 * result.white_center
        + 0.30 * result.white_total
        + 0.18 * min(1.0, result.blob_area_ratio / 0.72)
        + 0.07 * max(0.0, 1.0 - result.blob_offset)
    )
    uniform_background_penalty = max(0.0, result.white_total - 0.90) * 3.0 + max(0.0, result.blob_area_ratio - 0.90) * 2.0
    return base - uniform_background_penalty


def classify_side_candidates(
    side: str,
    frame_bgr: np.ndarray,
    rois: Sequence[Tuple[str, int, int, int, int, bool]],
    cfg: CapDetectorConfig,
) -> Tuple[EndResult, np.ndarray]:
    candidates: List[Tuple[EndResult, np.ndarray, str]] = []
    for roi in rois:
        if not roi[0].startswith(side + "_"):
            continue
        result, crop = classify_side_roi(roi[0], frame_bgr, roi, cfg)
        orientation = "right" if roi[0].endswith("_right") else "left"
        candidates.append((result, crop, orientation))
    if not candidates:
        return (
            EndResult(side, "UNKNOWN", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1.8, "missing_side_candidates"),
            np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=np.uint8),
        )

    visible_present = [
        item for item in candidates if item[0].state == "PRESENT" and item[0].visibility >= 0.80
    ]
    if visible_present:
        result, crop, orientation = max(visible_present, key=lambda item: side_candidate_score(item[0]))
        return relabel_side_result(result, side, f"{orientation}_side_beige_sleeve_visible"), crop

    visible_missing = [
        item for item in candidates if item[0].state == "MISSING" and item[0].visibility >= 0.80
    ]
    visible_unknown = [
        item for item in candidates if item[0].state == "UNKNOWN" and item[0].visibility >= 0.55
    ]
    if visible_unknown:
        result, crop, orientation = max(visible_unknown, key=lambda item: side_candidate_score(item[0]))
        return relabel_side_result(result, side, f"{orientation}_{result.reason}"), crop
    if visible_missing and len(visible_missing) == len([item for item in candidates if item[0].visibility >= 0.80]):
        result, crop, orientation = min(visible_missing, key=lambda item: side_candidate_score(item[0]))
        return relabel_side_result(result, side, f"{orientation}_side_beige_sleeve_absent"), crop

    result, crop, orientation = max(candidates, key=lambda item: (item[0].visibility, side_candidate_score(item[0])))
    return relabel_side_result(result, side, f"{orientation}_side_evidence_incomplete"), crop


def sample_state(top: EndResult, bottom: EndResult, side_upper: EndResult, side_lower: EndResult, paired: bool) -> Tuple[str, str, str]:
    if not paired:
        return "UNKNOWN", "", "single_anchor_no_pair"
    ends = [top, bottom, side_upper, side_lower]
    missing_results = [r for r in ends if r.state == "MISSING"]
    soft_port_missing = [
        r for r in missing_results if r.side in {"top", "bottom"} and r.reason == "weak_beige_evidence_at_port"
    ]
    port_missing = [r for r in missing_results if r.side in {"top", "bottom"}]
    side_missing = [r for r in missing_results if r.side.startswith("side_")]
    hard_missing = [r for r in port_missing if r not in soft_port_missing]
    missing = [r.side for r in missing_results]
    unknown = [r.reason for r in ends if r.state == "UNKNOWN"]
    if hard_missing or len(soft_port_missing) >= 2 or (soft_port_missing and side_missing):
        return "DEFECT", "+".join(missing), ""
    if soft_port_missing:
        unknown.extend([f"weak_{r.side}_port_evidence" for r in soft_port_missing])
    if side_missing:
        unknown.extend([f"{r.side}_needs_bilateral_confirmation" for r in side_missing])
    if unknown:
        return "UNKNOWN", "", ";".join(sorted(set(unknown)))
    if all(r.state == "PRESENT" for r in ends):
        return "NORMAL", "", ""
    return "UNKNOWN", "", "inconsistent_end_states"


def collect_pseudo_normal_crops(video_path: Path, cfg: CapDetectorConfig) -> List[np.ndarray]:
    cap, fps, frame_count, _, _ = open_video(video_path)
    duration = frame_count / fps if fps else 0.0
    end_sec = cfg.scan_end_sec if cfg.scan_end_sec is not None else duration
    crops: List[np.ndarray] = []
    for t in tqdm(list(iter_times(cfg.scan_start_sec, end_sec, cfg.train_step_sec)), desc="collect_pseudo_normal"):
        ok, frame, _ = read_frame_at(cap, t)
        if not ok or frame is None:
            continue
        pairs, _ = pair_anchors(detect_blue_anchors(frame, cfg), cfg)
        for top_anchor, bottom_anchor in pairs:
            for side, anchor in [("top", top_anchor), ("bottom", bottom_anchor)]:
                x1, y1, x2, y2, clipped = sleeve_roi(frame.shape, anchor, side, cfg)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                metrics = sleeve_metrics(crop, clipped)
                if (
                    not metrics.clipped
                    and metrics.blur_var >= cfg.min_blur_var
                    and metrics.white_center >= cfg.pseudo_normal_white_center_min
                    and metrics.white_total >= 0.60
                ):
                    crops.append(cv2.resize(crop, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(crops) > 1200:
        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(len(crops), size=1200, replace=False)
        crops = [crops[int(i)] for i in idx]
    return crops


def scan_video(
    video_path: Path,
    cfg: CapDetectorConfig,
    memory: CapPatchMemory,
) -> Tuple[List[CapObservation], Dict[Tuple[float, int], dict]]:
    cap, fps, frame_count, _, _ = open_video(video_path)
    duration = frame_count / fps if fps else 0.0
    end_sec = cfg.scan_end_sec if cfg.scan_end_sec is not None else duration
    observations: List[CapObservation] = []
    diagnostics: Dict[Tuple[float, int], dict] = {}
    for t in tqdm(list(iter_times(cfg.scan_start_sec, end_sec, cfg.sample_step_sec)), desc="scan_caps"):
        ok, frame, frame_idx = read_frame_at(cap, t)
        if not ok or frame is None:
            continue
        anchors = detect_blue_anchors(frame, cfg)
        pairs, unpaired = pair_anchors(anchors, cfg)
        sample_index = 0
        for top_anchor, bottom_anchor in pairs:
            top, top_crop, top_heat = classify_end("top", frame, top_anchor, cfg, memory)
            bottom, bottom_crop, bottom_heat = classify_end("bottom", frame, bottom_anchor, cfg, memory)
            side_rois = side_sleeve_candidate_rois(frame.shape, top_anchor, bottom_anchor, cfg)
            side_upper, side_upper_crop = classify_side_candidates("side_upper", frame, side_rois, cfg)
            side_lower, side_lower_crop = classify_side_candidates("side_lower", frame, side_rois, cfg)
            raw_state, missing_side, unknown_reason = sample_state(top, bottom, side_upper, side_lower, True)
            anomaly = max(top.anomaly_score, bottom.anomaly_score, side_upper.anomaly_score, side_lower.anomaly_score)
            side_state = "MISSING" if "side" in missing_side else ("UNKNOWN" if side_upper.state == "UNKNOWN" or side_lower.state == "UNKNOWN" else "PRESENT")
            sample_cx = float((top_anchor.cx + bottom_anchor.cx) / 2.0)
            sample_index += 1
            obs = CapObservation(
                time_sec=float(t),
                frame_idx=int(frame_idx),
                sample_index=sample_index,
                sample_cx=sample_cx,
                sample_y1=int(min(top_anchor.y, bottom_anchor.y)),
                sample_y2=int(max(top_anchor.y + top_anchor.h, bottom_anchor.y + bottom_anchor.h)),
                paired=True,
                top_anchor_x=top_anchor.x,
                top_anchor_y=top_anchor.y,
                top_anchor_w=top_anchor.w,
                top_anchor_h=top_anchor.h,
                bottom_anchor_x=bottom_anchor.x,
                bottom_anchor_y=bottom_anchor.y,
                bottom_anchor_w=bottom_anchor.w,
                bottom_anchor_h=bottom_anchor.h,
                top_state=top.state,
                bottom_state=bottom.state,
                side_upper_state=side_upper.state,
                side_lower_state=side_lower.state,
                side_state=side_state,
                raw_state=raw_state,
                final_state=raw_state,
                missing_side=missing_side,
                unknown_reason=unknown_reason,
                top_white_center=top.white_center,
                bottom_white_center=bottom.white_center,
                top_white_total=top.white_total,
                bottom_white_total=bottom.white_total,
                top_blue_center=top.blue_center,
                bottom_blue_center=bottom.blue_center,
                side_upper_white_center=side_upper.white_center,
                side_lower_white_center=side_lower.white_center,
                top_patch_score=top.patch_score,
                bottom_patch_score=bottom.patch_score,
                anomaly_score=float(anomaly),
                event_id=0,
            )
            observations.append(obs)
            diagnostics[(round(float(t), 4), sample_index)] = {
                "top": top,
                "bottom": bottom,
                "side_upper": side_upper,
                "side_lower": side_lower,
                "top_crop": top_crop,
                "bottom_crop": bottom_crop,
                "side_upper_crop": side_upper_crop,
                "side_lower_crop": side_lower_crop,
                "top_heat": top_heat,
                "bottom_heat": bottom_heat,
            }
        # Unpaired blue fragments are intentionally ignored in the main observations.
        # In this mirrored setup they are often labels, reflections, or edge fragments,
        # while reliable decisions require a top/bottom anchor pair on the right half.
    cap.release()
    return observations, diagnostics


def summarize_event(group: Sequence[CapObservation], event_id: int) -> dict:
    peak = max(group, key=lambda o: o.anomaly_score)
    return {
        "event_id": event_id,
        "start_sec": float(group[0].time_sec),
        "end_sec": float(group[-1].time_sec),
        "duration_sec": float(group[-1].time_sec - group[0].time_sec),
        "observations": len(group),
        "peak_time_sec": float(peak.time_sec),
        "peak_anomaly_score": float(peak.anomaly_score),
        "peak_sample_cx": float(peak.sample_cx),
        "peak_missing_side": peak.missing_side,
    }


def apply_event_filter(observations: Sequence[CapObservation], cfg: CapDetectorConfig) -> Tuple[List[CapObservation], List[dict]]:
    def strong_defect(obs: CapObservation) -> bool:
        missing_parts = [part for part in obs.missing_side.split("+") if part]
        return len(missing_parts) >= 2 or obs.anomaly_score >= 0.35

    defects = sorted([o for o in observations if o.raw_state == "DEFECT"], key=lambda o: (o.time_sec, o.sample_cx))
    groups: List[List[CapObservation]] = []
    current: List[CapObservation] = []
    for obs in defects:
        if current and obs.time_sec - current[-1].time_sec > cfg.event_gap_sec:
            if len(current) >= cfg.min_event_observations or any(strong_defect(o) for o in current):
                groups.append(current)
            current = []
        current.append(obs)
    if current and (len(current) >= cfg.min_event_observations or any(strong_defect(o) for o in current)):
        groups.append(current)
    event_ranges = []
    events = []
    for idx, group in enumerate(groups, start=1):
        event = summarize_event(group, idx)
        events.append(event)
        event_ranges.append((event["start_sec"], event["end_sec"], idx))
    filtered: List[CapObservation] = []
    for obs in observations:
        final_state = obs.raw_state
        event_id = 0
        if obs.raw_state == "DEFECT":
            match = next((eid for start, end, eid in event_ranges if start <= obs.time_sec <= end), 0)
            if match:
                event_id = int(match)
            else:
                final_state = "FILTERED_SPIKE"
        filtered.append(CapObservation(**{**asdict(obs), "final_state": final_state, "event_id": event_id}))
    return filtered, events


def save_rows(path: Path, rows: Sequence[CapObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CapObservation.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_events(path: Path, events: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["event_id", "start_sec", "end_sec", "duration_sec", "observations", "peak_time_sec", "peak_anomaly_score", "peak_sample_cx", "peak_missing_side"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for event in events:
            writer.writerow(event)


def save_model(path: Path, cfg: CapDetectorConfig, memory: CapPatchMemory, normal_scores: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "config": asdict(cfg),
                "scaler": memory.scaler,
                "memory": memory.memory,
                "threshold": memory.threshold,
                "normal_scores": list(map(float, normal_scores)),
            },
            f,
        )


def load_model(path: Path) -> Tuple[CapDetectorConfig, CapPatchMemory, List[float]]:
    with path.open("rb") as f:
        bundle = pickle.load(f)
    cfg_fields = set(CapDetectorConfig.__dataclass_fields__.keys())
    cfg = CapDetectorConfig(**{k: v for k, v in bundle["config"].items() if k in cfg_fields})
    memory = CapPatchMemory(cfg)
    memory.scaler = bundle["scaler"]
    memory.memory = bundle["memory"]
    memory.threshold = float(bundle["threshold"])
    memory.nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    memory.nn.fit(memory.memory)
    return cfg, memory, list(bundle.get("normal_scores", []))


def normalize_heat(heat: Optional[np.ndarray]) -> np.ndarray:
    if heat is None:
        return np.zeros((8, 8), dtype=np.float32)
    h = heat.astype(np.float32).copy()
    scale = max(float(np.percentile(h[h > 0], 95)) if np.any(h > 0) else 1.0, 1e-6)
    return np.clip(h / scale, 0, 1)


def heat_overlay(crop: np.ndarray, heat: Optional[np.ndarray]) -> np.ndarray:
    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    h = cv2.resize(normalize_heat(heat), (96, 96), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(crop, 0.52, color, 0.48, 0)


def state_color(state: str) -> Color:
    if state == "DEFECT" or state == "MISSING":
        return (0, 60, 255)
    if state == "FILTERED_SPIKE":
        return (0, 165, 255)
    if state == "UNKNOWN":
        return (0, 230, 255)
    return (60, 220, 80)


def draw_label(img: np.ndarray, text: str, xy: Tuple[int, int], color: Color, scale: float = 0.62) -> None:
    x, y = xy
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 8, y + 6), (10, 13, 20), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_metric_bar(img: np.ndarray, label: str, value: float, vmax: float, x0: int, y0: int, x1: int, color: Color) -> None:
    cv2.putText(img, f"{label.upper()} {value:.3f}", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 240, 245), 1)
    cv2.rectangle(img, (x0, y0), (x1, y0 + 17), (42, 48, 62), -1)
    frac = min(1.0, max(0.0, value / max(vmax, 1e-6)))
    cv2.rectangle(img, (x0, y0), (int(x0 + (x1 - x0) * frac), y0 + 17), color, -1)


def annotate_frame(
    frame: np.ndarray,
    observations: Sequence[CapObservation],
    diagnostics: Dict[Tuple[float, int], dict],
    cfg: CapDetectorConfig,
    time_sec: float,
    recent: Sequence[CapObservation],
) -> np.ndarray:
    canvas = frame.copy()
    veil = canvas.copy()
    cv2.rectangle(veil, (0, 0), (cfg.right_half_x, frame.shape[0]), (0, 0, 0), -1)
    cv2.rectangle(veil, (cfg.right_half_x, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (15, 18, 26), 2)
    canvas = cv2.addWeighted(veil, 0.34, canvas, 0.66, 0)
    cv2.putText(canvas, "RIGHT HALF ONLY - BEIGE SLEEVE INSPECTION", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (240, 248, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"t={time_sec:07.2f}s  blue=locator only  target=beige sleeve", (28, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 225, 240), 2, cv2.LINE_AA)
    cv2.line(canvas, (cfg.right_half_x, 0), (cfg.right_half_x, frame.shape[0]), (0, 230, 255), 3)

    selected_diag = None
    selected_obs = None
    for obs in observations:
        color = state_color(obs.final_state)
        if obs.paired:
            cv2.rectangle(canvas, (obs.top_anchor_x, obs.top_anchor_y), (obs.top_anchor_x + obs.top_anchor_w, obs.top_anchor_y + obs.top_anchor_h), (255, 180, 0), 2)
            cv2.rectangle(canvas, (obs.bottom_anchor_x, obs.bottom_anchor_y), (obs.bottom_anchor_x + obs.bottom_anchor_w, obs.bottom_anchor_y + obs.bottom_anchor_h), (255, 180, 0), 2)
            cv2.line(canvas, (int(obs.sample_cx), obs.top_anchor_y), (int(obs.sample_cx), obs.bottom_anchor_y + obs.bottom_anchor_h), color, 2)
            diag = diagnostics.get((round(obs.time_sec, 4), obs.sample_index))
            if diag:
                for key in ["top", "bottom", "side_upper", "side_lower"]:
                    end: EndResult = diag[key]
                    cv2.rectangle(canvas, (end.roi_x1, end.roi_y1), (end.roi_x2, end.roi_y2), state_color(end.state), 3)
                    cv2.putText(canvas, f"{end.side.upper()} {end.state}", (end.roi_x1, max(18, end.roi_y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, state_color(end.state), 2, cv2.LINE_AA)
            label = f"{obs.final_state}  top={obs.top_state} bottom={obs.bottom_state} side={obs.side_state}"
            draw_label(canvas, label, (int(obs.sample_cx) - 170, max(105, obs.sample_y1 - 18)), color, 0.55)
        else:
            cv2.rectangle(canvas, (obs.top_anchor_x, obs.top_anchor_y), (obs.top_anchor_x + obs.top_anchor_w, obs.top_anchor_y + obs.top_anchor_h), state_color("UNKNOWN"), 2)
            draw_label(canvas, "UNKNOWN single anchor", (obs.top_anchor_x, max(105, obs.top_anchor_y - 12)), state_color("UNKNOWN"), 0.5)
        if selected_obs is None or obs.final_state == "DEFECT" or (selected_obs.final_state not in ["DEFECT", "UNKNOWN"] and obs.final_state == "UNKNOWN"):
            selected_obs = obs
            selected_diag = diagnostics.get((round(obs.time_sec, 4), obs.sample_index))

    x0, y0, x1, y1 = 26, 96, 526, 1010
    panel = canvas.copy()
    cv2.rectangle(panel, (x0, y0), (x1, y1), (10, 13, 20), -1)
    canvas = cv2.addWeighted(panel, 0.76, canvas, 0.24, 0)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), state_color(selected_obs.final_state if selected_obs else "UNKNOWN"), 2)
    cv2.putText(canvas, "SLEEVE DIAGNOSTICS", (x0 + 24, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 250, 255), 2)

    if selected_obs is not None and selected_diag and selected_obs.paired:
        for idx, key in enumerate(["top", "bottom"]):
            end: EndResult = selected_diag[key]
            crop = selected_diag[f"{key}_crop"]
            heat = selected_diag[f"{key}_heat"]
            yy = y0 + 70 + idx * 250
            crop_small = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
            heat_small = cv2.resize(heat_overlay(crop, heat), (112, 112), interpolation=cv2.INTER_AREA)
            canvas[yy : yy + 112, x0 + 28 : x0 + 140] = crop_small
            canvas[yy : yy + 112, x0 + 160 : x0 + 272] = heat_small
            cv2.putText(canvas, f"{key.upper()} {end.state}", (x0 + 292, yy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, state_color(end.state), 2)
            cv2.putText(canvas, end.reason[:24], (x0 + 292, yy + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 225, 240), 1)
            draw_metric_bar(canvas, "white", end.white_center, 1.0, x0 + 292, yy + 88, x0 + 462, (230, 230, 210))
            draw_metric_bar(canvas, "blue leak", end.blue_center, 0.8, x0 + 292, yy + 138, x0 + 462, (255, 180, 0))
            draw_metric_bar(canvas, "anomaly", end.anomaly_score, 2.2, x0 + 292, yy + 188, x0 + 462, state_color(end.state))
        sy = y0 + 575
        for sidx, key in enumerate(["side_upper", "side_lower"]):
            end = selected_diag[key]
            cv2.putText(canvas, f"{key.upper()} {end.state}", (x0 + 34, sy + sidx * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color(end.state), 2)
            cv2.putText(canvas, f"white={end.white_center:.3f}  {end.reason[:18]}", (x0 + 250, sy + sidx * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 225, 240), 1)
    else:
        cv2.putText(canvas, "No paired sample in this frame.", (x0 + 34, y0 + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 225, 240), 2)

    strip = (x0 + 34, y0 + 650, x0 + 462, y0 + 842)
    cv2.putText(canvas, "RECENT MAX ANOMALY", (strip[0], strip[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 245), 1)
    cv2.rectangle(canvas, (strip[0], strip[1]), (strip[2], strip[3]), (24, 28, 38), -1)
    cv2.rectangle(canvas, (strip[0], strip[1]), (strip[2], strip[3]), (85, 96, 120), 1)
    if recent:
        by_time: Dict[float, float] = {}
        for obs in recent[-120:]:
            by_time[obs.time_sec] = max(by_time.get(obs.time_sec, 0.0), obs.anomaly_score)
        vals = list(by_time.items())[-80:]
        vmax = max([v for _, v in vals] + [1.6])
        pts = []
        for idx, (_, val) in enumerate(vals):
            px = int(strip[0] + (strip[2] - strip[0]) * idx / max(1, len(vals) - 1))
            py = int(strip[3] - (strip[3] - strip[1]) * min(1.0, val / vmax))
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, a, b, (0, 220, 255), 2)
    return canvas


def save_blue_locator(video_path: Path, cfg: CapDetectorConfig, out_path: Path, time_sec: float = 20.0) -> None:
    cap, _, _, _, _ = open_video(video_path)
    ok, frame, _ = read_frame_at(cap, time_sec)
    cap.release()
    if not ok or frame is None:
        return
    anchors = detect_blue_anchors(frame, cfg)
    pairs, unpaired = pair_anchors(anchors, cfg)
    vis = frame.copy()
    cv2.rectangle(vis, (0, 0), (cfg.right_half_x, vis.shape[0]), (0, 0, 0), -1)
    cv2.line(vis, (cfg.right_half_x, 0), (cfg.right_half_x, vis.shape[0]), (0, 230, 255), 3)
    for top, bottom in pairs:
        cv2.rectangle(vis, (top.x, top.y), (top.x + top.w, top.y + top.h), (255, 180, 0), 3)
        cv2.rectangle(vis, (bottom.x, bottom.y), (bottom.x + bottom.w, bottom.y + bottom.h), (255, 180, 0), 3)
        cv2.line(vis, (int(top.cx), int(top.cy)), (int(bottom.cx), int(bottom.cy)), (0, 220, 80), 2)
        for side, anchor in [("top", top), ("bottom", bottom)]:
            x1, y1, x2, y2, _ = sleeve_roi(vis.shape, anchor, side, cfg)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 255), 3)
            cv2.putText(vis, "BEIGE SLEEVE ROI", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2)
        for name, x1, y1, x2, y2, _ in side_sleeve_candidate_rois(vis.shape, top, bottom, cfg):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(vis, name.upper(), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2)
    for anchor in unpaired:
        cv2.rectangle(vis, (anchor.x, anchor.y), (anchor.x + anchor.w, anchor.y + anchor.h), (0, 230, 255), 2)
    draw_label(vis, "blue boxes are locator anchors; yellow boxes are target beige sleeve ROIs", (cfg.right_half_x + 30, 54), (0, 230, 255), 0.72)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def save_training_mosaic(crops: Sequence[np.ndarray], out_path: Path) -> None:
    if not crops:
        return
    rng = np.random.default_rng(7)
    idx = rng.choice(len(crops), size=min(80, len(crops)), replace=False)
    tile = 96
    cols = 10
    rows = int(math.ceil(len(idx) / cols))
    header = 80
    img = Image.new("RGB", (cols * tile, rows * tile + header), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    draw.text((22, 22), "Pseudo-Normal Beige Sleeve Training Crops", fill=(235, 245, 255), font=pil_font(24))
    for n, i in enumerate(idx):
        crop = cv2.cvtColor(cv2.resize(crops[int(i)], (tile, tile)), cv2.COLOR_BGR2RGB)
        x = (n % cols) * tile
        y = header + (n // cols) * tile
        img.paste(Image.fromarray(crop), (x, y))
        ImageDraw.Draw(img).rectangle((x, y, x + tile - 1, y + tile - 1), outline=(0, 230, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_timeline(observations: Sequence[CapObservation], events: Sequence[dict], out_path: Path) -> None:
    width, height = 1800, 640
    img = Image.new("RGB", (width, height), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    draw.text((34, 26), "Cap Sleeve Detection Timeline", fill=(235, 245, 255), font=pil_font(31))
    if not observations:
        img.save(out_path)
        return
    by_time: Dict[float, dict] = {}
    for obs in observations:
        row = by_time.setdefault(obs.time_sec, {"score": 0.0, "defect": 0, "unknown": 0, "filtered": 0})
        row["score"] = max(row["score"], obs.anomaly_score)
        row["defect"] += int(obs.final_state == "DEFECT")
        row["unknown"] += int(obs.final_state == "UNKNOWN")
        row["filtered"] += int(obs.final_state == "FILTERED_SPIKE")
    xs = np.asarray(sorted(by_time.keys()), dtype=np.float32)
    ys = np.asarray([by_time[float(t)]["score"] for t in xs], dtype=np.float32)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_max = max(float(ys.max()), 1.6)
    plot = (80, 120, width - 60, height - 90)
    draw.rectangle(plot, outline=(88, 105, 130), width=2)
    for event in events:
        sx = plot[0] + int((event["start_sec"] - x_min) / max(1e-6, x_max - x_min) * (plot[2] - plot[0]))
        ex = plot[0] + int((event["end_sec"] - x_min) / max(1e-6, x_max - x_min) * (plot[2] - plot[0]))
        draw.rectangle((sx, plot[1], ex, plot[3]), fill=(82, 26, 28))
    pts = []
    for xval, yval in zip(xs, ys):
        x = plot[0] + int((float(xval) - x_min) / max(1e-6, x_max - x_min) * (plot[2] - plot[0]))
        y = plot[3] - int(min(1.0, float(yval) / y_max) * (plot[3] - plot[1]))
        pts.append((x, y))
    for a, b in zip(pts[:-1], pts[1:]):
        draw.line((a[0], a[1], b[0], b[1]), fill=(0, 220, 255), width=2)
    for t, point in zip(xs, pts):
        row = by_time[float(t)]
        if row["defect"]:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(255, 60, 58))
        elif row["unknown"]:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(255, 230, 0))
        elif row["filtered"]:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=(255, 165, 0))
    draw.text((plot[0], plot[3] + 28), f"{x_min:.1f}s", fill=(190, 205, 220), font=pil_font(17))
    draw.text((plot[2] - 90, plot[3] + 28), f"{x_max:.1f}s", fill=(190, 205, 220), font=pil_font(17))
    draw.text((plot[0] + 10, plot[1] + 10), "cyan=max anomaly  red=defect event  yellow=unknown  orange=filtered spike", fill=(220, 230, 240), font=pil_font(18))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def build_gallery_items(observations: Sequence[CapObservation]) -> List[CapObservation]:
    defects = sorted([o for o in observations if o.final_state == "DEFECT"], key=lambda o: o.anomaly_score, reverse=True)
    unknowns = sorted([o for o in observations if o.final_state == "UNKNOWN"], key=lambda o: o.time_sec)
    filtered = sorted([o for o in observations if o.final_state == "FILTERED_SPIKE"], key=lambda o: o.anomaly_score, reverse=True)
    normals = sorted([o for o in observations if o.final_state == "NORMAL"], key=lambda o: abs(o.time_sec - 20.0))
    selected: List[CapObservation] = []
    for pool, limit in [(defects, 8), (unknowns, 6), (filtered, 4), (normals, 4)]:
        for item in pool[:limit]:
            selected.append(item)
    seen = set()
    unique = []
    for item in selected:
        key = (round(item.time_sec, 2), item.sample_index)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique[:20]


def save_gallery(video_path: Path, cfg: CapDetectorConfig, memory: CapPatchMemory, observations: Sequence[CapObservation], out_path: Path) -> None:
    items = build_gallery_items(observations)
    if not items:
        return
    cap, _, _, _, _ = open_video(video_path)
    cols = 4
    tile_w, tile_h = 420, 360
    rows = int(math.ceil(len(items) / cols))
    img = Image.new("RGB", (cols * tile_w + 40, rows * tile_h + 92), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    draw.text((28, 24), "Defect / Unknown / Filtered Spike Gallery", fill=(235, 245, 255), font=pil_font(28))
    for idx, obs in enumerate(items):
        ok, frame, _ = read_frame_at(cap, obs.time_sec)
        if not ok or frame is None:
            continue
        anchors = detect_blue_anchors(frame, cfg)
        pairs, _ = pair_anchors(anchors, cfg)
        matched = None
        for pidx, pair in enumerate(pairs, start=1):
            if pidx == obs.sample_index:
                matched = pair
                break
        crops = []
        if matched is not None:
            for side, anchor in [("top", matched[0]), ("bottom", matched[1])]:
                end, crop, heat = classify_end(side, frame, anchor, cfg, memory)
                crops.append((end, crop, heat_overlay(crop, heat)))
        x = 20 + (idx % cols) * tile_w
        y = 92 + (idx // cols) * tile_h
        accent = state_color(obs.final_state)
        draw.rounded_rectangle((x, y, x + tile_w - 18, y + tile_h - 18), radius=8, fill=(18, 23, 33), outline=accent[::-1] if False else tuple(reversed(accent)), width=2)
        draw.text((x + 18, y + 14), f"t={obs.time_sec:.2f}s  {obs.final_state}", fill=(accent[2], accent[1], accent[0]), font=pil_font(22))
        for cidx, (_, crop, overlay) in enumerate(crops[:2]):
            px = x + 18 + cidx * 190
            py = y + 52
            pair_img = np.hstack([cv2.resize(crop, (84, 84)), cv2.resize(overlay, (84, 84))])
            img.paste(Image.fromarray(cv2.cvtColor(pair_img, cv2.COLOR_BGR2RGB)), (px, py))
        lines = [
            f"top/bottom/side: {obs.top_state} / {obs.bottom_state} / {obs.side_state}",
            f"white center: {obs.top_white_center:.2f} / {obs.bottom_white_center:.2f}",
            f"side white: {obs.side_upper_white_center:.2f} / {obs.side_lower_white_center:.2f}",
            f"blue leak: {obs.top_blue_center:.2f} / {obs.bottom_blue_center:.2f}",
            f"anomaly: {obs.anomaly_score:.3f}",
            f"reason: {obs.unknown_reason or obs.missing_side or 'ok'}",
        ]
        for li, line in enumerate(lines):
            draw.text((x + 18, y + 222 + li * 21), line, fill=(210, 222, 235), font=pil_font(14))
    cap.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_dashboard(observations: Sequence[CapObservation], normal_scores: Sequence[float], memory: CapPatchMemory, out_path: Path) -> None:
    width, height = 1800, 980
    img = Image.new("RGB", (width, height), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    draw.text((34, 26), "Beige Sleeve Evidence Dashboard", fill=(235, 245, 255), font=pil_font(32))
    states = ["NORMAL", "DEFECT", "UNKNOWN", "FILTERED_SPIKE"]
    colors = {"NORMAL": (68, 224, 95), "DEFECT": (255, 64, 58), "UNKNOWN": (255, 225, 45), "FILTERED_SPIKE": (255, 148, 42)}
    counts = {s: sum(o.final_state == s for o in observations) for s in states}
    for idx, s in enumerate(states):
        x = 60 + idx * 410
        y = 92
        draw.rounded_rectangle((x, y, x + 350, y + 112), radius=8, fill=(24, 29, 41), outline=colors[s], width=2)
        draw.text((x + 18, y + 17), s, fill=(180, 194, 212), font=pil_font(16))
        draw.text((x + 18, y + 48), str(counts[s]), fill=colors[s], font=pil_font(34))
    plot = (80, 280, 820, 650)
    draw.rectangle(plot, outline=(88, 105, 130), width=2)
    draw.text((plot[0], plot[1] - 38), "White-center ratio distribution", fill=(235, 245, 255), font=pil_font(23))
    bins = np.linspace(0, 1, 31)
    max_count = 1
    hist_by_state = {}
    for s in states:
        vals = []
        for o in observations:
            if o.final_state == s:
                vals.extend([o.top_white_center, o.bottom_white_center])
        hist, _ = np.histogram(vals, bins=bins)
        hist_by_state[s] = hist
        max_count = max(max_count, int(hist.max()) if hist.size else 1)
    bar_w = max(2, (plot[2] - plot[0]) // (len(bins) - 1))
    for sidx, s in enumerate(states):
        hist = hist_by_state[s]
        for i, c in enumerate(hist):
            h = int(c / max_count * (plot[3] - plot[1] - 36))
            x = plot[0] + i * bar_w + sidx * max(1, bar_w // 5)
            draw.rectangle((x, plot[3] - h, x + max(1, bar_w // 5), plot[3]), fill=colors[s])
    for idx, s in enumerate(states):
        lx = plot[0] + idx * 170
        draw.rectangle((lx, plot[1] + 16, lx + 18, plot[1] + 34), fill=colors[s])
        draw.text((lx + 26, plot[1] + 12), s, fill=(230, 240, 250), font=pil_font(14))
    plot2 = (980, 280, 1700, 650)
    draw.rectangle(plot2, outline=(88, 105, 130), width=2)
    draw.text((plot2[0], plot2[1] - 38), "Patch memory normal-score envelope", fill=(235, 245, 255), font=pil_font(23))
    if normal_scores:
        vals = np.asarray(normal_scores, dtype=np.float64)
        hist, edges = np.histogram(vals, bins=32)
        ymax = max(1, int(hist.max()))
        for i, c in enumerate(hist):
            h = int(c / ymax * (plot2[3] - plot2[1] - 40))
            x0 = plot2[0] + int(i / len(hist) * (plot2[2] - plot2[0]))
            x1 = plot2[0] + int((i + 1) / len(hist) * (plot2[2] - plot2[0])) - 1
            draw.rectangle((x0, plot2[3] - h, x1, plot2[3]), fill=(0, 210, 255))
        thx = plot2[0] + int((memory.threshold - float(edges[0])) / max(1e-6, float(edges[-1] - edges[0])) * (plot2[2] - plot2[0]))
        draw.line((thx, plot2[1], thx, plot2[3]), fill=(255, 255, 255), width=2)
        draw.text((thx + 6, plot2[1] + 18), f"T={memory.threshold:.3f}", fill=(255, 255, 255), font=pil_font(16))
    draw.text((80, 760), "Rule: missing = weak beige sleeve evidence plus exposed blue/abnormal patch evidence. Unknown samples are isolated for review.", fill=(210, 222, 235), font=pil_font(22))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_review_video(
    video_path: Path,
    cfg: CapDetectorConfig,
    memory: CapPatchMemory,
    observations: Sequence[CapObservation],
    diagnostics: Dict[Tuple[float, int], dict],
    out_path: Path,
) -> None:
    cap, _, _, width, height = open_video(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.review_fps, (width, height))
    obs_by_time: Dict[float, List[CapObservation]] = {}
    for obs in observations:
        obs_by_time.setdefault(round(obs.time_sec, 4), []).append(obs)
    recent: List[CapObservation] = []
    if observations:
        start = min(o.time_sec for o in observations)
        end = max(o.time_sec for o in observations)
    else:
        start, end = cfg.scan_start_sec, cfg.scan_start_sec
    for t in tqdm(list(iter_times(start, end, cfg.review_step_sec)), desc="review_video"):
        ok, frame, _ = read_frame_at(cap, t)
        if not ok or frame is None:
            continue
        rows = obs_by_time.get(round(t, 4), [])
        recent.extend(rows)
        writer.write(annotate_frame(frame, rows, diagnostics, cfg, t, recent))
    writer.release()
    cap.release()


def run_pipeline(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    out_dir = Path(args.out)
    cfg = CapDetectorConfig(
        sample_step_sec=args.sample_step,
        train_step_sec=args.train_step,
        scan_start_sec=args.scan_start,
        scan_end_sec=args.scan_end,
        review_fps=args.review_fps,
        review_step_sec=args.sample_step,
    )
    train_crops = collect_pseudo_normal_crops(video_path, cfg)
    memory = CapPatchMemory(cfg)
    normal_scores = memory.fit(train_crops)
    observations, diagnostics = scan_video(video_path, cfg, memory)
    observations, events = apply_event_filter(observations, cfg)

    save_rows(out_dir / "observations.csv", observations)
    save_rows(out_dir / "unknown_samples.csv", [o for o in observations if o.final_state == "UNKNOWN"])
    save_events(out_dir / "defect_events.csv", events)
    save_model(out_dir / "model" / "cap_patch_memory.pkl", cfg, memory, normal_scores)
    save_blue_locator(video_path, cfg, out_dir / "visuals" / "01_blue_cap_locator.jpg")
    save_training_mosaic(train_crops, out_dir / "visuals" / "02_pseudo_normal_training_set.jpg")
    save_timeline(observations, events, out_dir / "visuals" / "03_score_timeline.png")
    save_gallery(video_path, cfg, memory, observations, out_dir / "visuals" / "04_defect_unknown_gallery.jpg")
    save_dashboard(observations, normal_scores, memory, out_dir / "visuals" / "05_cap_evidence_dashboard.png")
    save_review_video(video_path, cfg, memory, observations, diagnostics, out_dir / "visuals" / "06_review_2fps.mp4")

    counts = {state: int(sum(o.final_state == state for o in observations)) for state in ["NORMAL", "DEFECT", "UNKNOWN", "FILTERED_SPIKE"]}
    summary = {
        "video": str(video_path),
        "right_half_x": cfg.right_half_x,
        "pseudo_normal_crops": len(train_crops),
        "normal_score_threshold": memory.threshold,
        "observations": len(observations),
        "counts": counts,
        "defect_events": events,
        "outputs": {
            "observations_csv": str(out_dir / "observations.csv"),
            "unknown_samples_csv": str(out_dir / "unknown_samples.csv"),
            "defect_events_csv": str(out_dir / "defect_events.csv"),
            "visuals_dir": str(out_dir / "visuals"),
            "model": str(out_dir / "model" / "cap_patch_memory.pkl"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Right-half beige sleeve cap defect detector.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Train pseudo-normal sleeve memory, scan video, and create diagnostics.")
    run.add_argument("--video", default="20241101_161258.mp4")
    run.add_argument("--out", default="cap_defect_outputs")
    run.add_argument("--sample-step", type=float, default=0.5)
    run.add_argument("--train-step", type=float, default=2.0)
    run.add_argument("--scan-start", type=float, default=0.0)
    run.add_argument("--scan-end", type=float, default=None)
    run.add_argument("--review-fps", type=float, default=2.0)
    run.set_defaults(func=run_pipeline)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
