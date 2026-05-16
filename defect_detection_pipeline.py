from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


@dataclass
class DetectorConfig:
    gate_x: int = 900
    roi_x1: int = 600
    roi_y1: int = 280
    roi_x2: int = 1180
    roi_y2: int = 720
    min_radius: int = 65
    max_radius: int = 135
    crop_size: int = 224
    grid: int = 14
    inner_radius_px: int = 72
    max_memory_patches: int = 4096
    random_seed: int = 7
    train_start_sec: float = 0.0
    train_end_sec: float = 350.0
    val_start_sec: float = 350.0
    val_end_sec: float = 360.0
    scan_start_sec: float = 0.0
    scan_end_sec: Optional[float] = None
    train_step_sec: float = 1.0
    scan_step_sec: float = 0.25
    preview_start_sec: float = 360.0
    preview_end_sec: float = 430.0
    preview_step_sec: float = 0.20
    threshold_quantile: float = 0.995
    minimum_operational_threshold: float = 4.0
    min_detection_score: float = 0.42
    defect_min_gap_sec: float = 1.25
    min_event_observations: int = 2


@dataclass
class CircleCandidate:
    cx: int
    cy: int
    r: int
    detection_score: float
    yellow_ratio: float
    saturation: float
    value: float


@dataclass
class Observation:
    time_sec: float
    frame_idx: int
    cx: int
    cy: int
    r: int
    detection_score: float
    yellow_ratio: float
    saturation: float
    value: float
    patchcore_score: float
    hybrid_score: float
    is_defect: bool


class GateCircleDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg

    def detect(self, frame_bgr: np.ndarray) -> Tuple[Optional[CircleCandidate], List[CircleCandidate]]:
        cfg = self.cfg
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        roi = gray[cfg.roi_y1 : cfg.roi_y2, cfg.roi_x1 : cfg.roi_x2]
        blur = cv2.medianBlur(roi, 5)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=110,
            param1=80,
            param2=26,
            minRadius=cfg.min_radius,
            maxRadius=cfg.max_radius,
        )
        if circles is None:
            return None, []

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        yy, xx = np.ogrid[: frame_bgr.shape[0], : frame_bgr.shape[1]]
        candidates: List[CircleCandidate] = []
        for x, y, r in np.around(circles[0]).astype(int):
            cx = int(x + cfg.roi_x1)
            cy = int(y + cfg.roi_y1)
            r = int(r)
            if not (660 <= cx <= 1120 and 330 <= cy <= 680 and cfg.min_radius <= r <= cfg.max_radius):
                continue

            inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.55 * r) ** 2
            patch = hsv[inner]
            if patch.size == 0:
                continue

            h = patch[:, 0]
            s = patch[:, 1]
            v = patch[:, 2]
            yellow_ratio = float(((h >= 12) & (h <= 38) & (s >= 55) & (v >= 60)).mean())
            saturation = float(s.mean() / 255.0)
            value = float(v.mean() / 255.0)

            score = (
                0.70 * yellow_ratio
                + 0.20 * float(value > 0.35)
                + 0.10 * (r / max(1, cfg.max_radius))
                - 0.0018 * abs(cx - cfg.gate_x)
                - 0.0010 * abs(cy - 470)
            )
            if score < cfg.min_detection_score:
                continue

            candidates.append(
                CircleCandidate(
                    cx=cx,
                    cy=cy,
                    r=r,
                    detection_score=float(score),
                    yellow_ratio=yellow_ratio,
                    saturation=saturation,
                    value=value,
                )
            )

        candidates.sort(key=lambda c: c.detection_score, reverse=True)
        return (candidates[0] if candidates else None), candidates


class TexturePatchCore:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.nn: Optional[NearestNeighbors] = None
        self.memory: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.patch_mask = self._build_patch_mask()

    def _build_patch_mask(self) -> np.ndarray:
        cfg = self.cfg
        cell = cfg.crop_size / cfg.grid
        centers = []
        for gy in range(cfg.grid):
            for gx in range(cfg.grid):
                centers.append(((gx + 0.5) * cell, (gy + 0.5) * cell))
        centers_arr = np.asarray(centers, dtype=np.float32)
        center = cfg.crop_size / 2.0
        dist = np.sqrt(((centers_arr - center) ** 2).sum(axis=1))
        return dist <= cfg.inner_radius_px

    def crop_circle(self, frame_bgr: np.ndarray, cand: CircleCandidate) -> np.ndarray:
        cfg = self.cfg
        margin = int(cand.r * 1.22)
        x1 = max(0, cand.cx - margin)
        y1 = max(0, cand.cy - margin)
        x2 = min(frame_bgr.shape[1], cand.cx + margin)
        y2 = min(frame_bgr.shape[0], cand.cy + margin)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=np.uint8)
        return cv2.resize(crop, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)

    def _extract_patch_features(self, crop_bgr: np.ndarray) -> np.ndarray:
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
                hsvp = hsv[y1:y2, x1:x2]
                labp = lab[y1:y2, x1:x2]
                grayp = gray[y1:y2, x1:x2]
                gradp = grad[y1:y2, x1:x2]
                lapp = lap[y1:y2, x1:x2]

                h = hsvp[:, :, 0] / 180.0
                s = hsvp[:, :, 1] / 255.0
                v = hsvp[:, :, 2] / 255.0
                yellow = ((hsvp[:, :, 0] >= 12) & (hsvp[:, :, 0] <= 38) & (hsvp[:, :, 1] >= 70)).astype(
                    np.float32
                )
                hue_angle = 2 * np.pi * h
                feat = [
                    float(np.sin(hue_angle).mean()),
                    float(np.cos(hue_angle).mean()),
                    float(s.mean()),
                    float(s.std()),
                    float(v.mean()),
                    float(v.std()),
                    float((labp[:, :, 0] / 255.0).mean()),
                    float((labp[:, :, 1] / 255.0).mean()),
                    float((labp[:, :, 2] / 255.0).mean()),
                    float((labp[:, :, 1] / 255.0).std()),
                    float((labp[:, :, 2] / 255.0).std()),
                    float((gradp / 255.0).mean()),
                    float((gradp / 255.0).std()),
                    float((np.abs(lapp) / 255.0).mean()),
                    float((grayp / 255.0).std()),
                    float(yellow.mean()),
                    float(np.percentile(s, 75)),
                    float(np.percentile(v, 25)),
                ]
                feats.append(feat)
        feats_arr = np.asarray(feats, dtype=np.float32)
        return feats_arr[self.patch_mask]

    def fit(self, crops_bgr: Sequence[np.ndarray]) -> None:
        features = [self._extract_patch_features(crop) for crop in crops_bgr]
        patch_features = np.vstack(features)
        scaled = self.scaler.fit_transform(patch_features)
        if scaled.shape[0] > self.cfg.max_memory_patches:
            kmeans = MiniBatchKMeans(
                n_clusters=self.cfg.max_memory_patches,
                random_state=self.cfg.random_seed,
                batch_size=4096,
                n_init="auto",
                reassignment_ratio=0.01,
                verbose=0,
            )
            kmeans.fit(scaled)
            memory = kmeans.cluster_centers_.astype(np.float32)
        else:
            memory = scaled.astype(np.float32)
        self.memory = memory
        self.nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        self.nn.fit(memory)

    def score(self, crop_bgr: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        if self.nn is None:
            raise RuntimeError("Model is not fitted.")
        feats = self._extract_patch_features(crop_bgr)
        scaled = self.scaler.transform(feats)
        distances, _ = self.nn.kneighbors(scaled, return_distance=True)
        patch_scores = distances[:, 0].astype(np.float32)
        top_k = max(1, int(math.ceil(len(patch_scores) * 0.08)))
        image_score = float(np.sort(patch_scores)[-top_k:].mean())
        heat = np.zeros((self.cfg.grid * self.cfg.grid,), dtype=np.float32)
        heat[self.patch_mask] = patch_scores
        heat = heat.reshape(self.cfg.grid, self.cfg.grid)
        return image_score, heat, patch_scores


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


def inner_face_metrics(crop_bgr: np.ndarray) -> Tuple[float, float, float]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= (min(h, w) * 0.32) ** 2
    pix = hsv[mask]
    if pix.size == 0:
        return 0.0, 0.0, 0.0
    yellow = float(((pix[:, 0] >= 12) & (pix[:, 0] <= 38) & (pix[:, 1] >= 70) & (pix[:, 2] >= 50)).mean())
    saturation = float(pix[:, 1].mean() / 255.0)
    texture = float(cv2.Laplacian(gray, cv2.CV_32F)[mask].std() / 255.0)
    return yellow, saturation, texture


def hybrid_from_crop(patch_score: float, crop_bgr: np.ndarray) -> Tuple[float, float, float, float]:
    yellow, sat, texture = inner_face_metrics(crop_bgr)
    # The learned PatchCore score is primary. Saturation/texture only gives a conservative boost
    # for the very specific "yellow granular end face disappeared" failure mode.
    yellow_deficit = max(0.0, 0.50 - sat) * 4.0 + max(0.0, 0.18 - texture) * 2.0
    return float(patch_score + yellow_deficit), yellow, sat, texture


def collect_crops(
    video_path: Path,
    cfg: DetectorConfig,
    start_sec: float,
    end_sec: float,
    step_sec: float,
    label: str,
) -> Tuple[List[np.ndarray], List[Tuple[float, CircleCandidate]]]:
    cap, _, _, _, _ = open_video(video_path)
    detector = GateCircleDetector(cfg)
    crops: List[np.ndarray] = []
    meta: List[Tuple[float, CircleCandidate]] = []
    model_helper = TexturePatchCore(cfg)
    times = list(iter_times(start_sec, end_sec, step_sec))
    for t in tqdm(times, desc=label):
        ok, frame, _ = read_frame_at(cap, t)
        if not ok or frame is None:
            continue
        cand, _ = detector.detect(frame)
        if cand is None:
            continue
        crop = model_helper.crop_circle(frame, cand)
        crops.append(crop)
        meta.append((t, cand))
    cap.release()
    return crops, meta


def set_threshold(model: TexturePatchCore, train_crops: Sequence[np.ndarray], val_crops: Sequence[np.ndarray], cfg: DetectorConfig) -> Tuple[float, List[float]]:
    normal_scores = []
    for crop in list(train_crops)[:: max(1, len(train_crops) // 120)]:
        score, _, _ = model.score(crop)
        hybrid, _, _, _ = hybrid_from_crop(score, crop)
        normal_scores.append(hybrid)
    for crop in val_crops:
        score, _, _ = model.score(crop)
        hybrid, _, _, _ = hybrid_from_crop(score, crop)
        normal_scores.append(hybrid)
    if not normal_scores:
        raise RuntimeError("No normal scores available for threshold calibration.")
    median = float(np.median(normal_scores))
    mad = float(np.median(np.abs(np.asarray(normal_scores) - median)) + 1e-6)
    # The calibration set can contain rare localization/lighting outliers. Using the 99.5%
    # quantile alone made this video miss obvious uncapped samples, so use a robust normal
    # envelope plus a fixed operating floor for the current score scale.
    robust_quantile = float(np.quantile(np.asarray(normal_scores), 0.975))
    threshold = max(cfg.minimum_operational_threshold, robust_quantile, median + 8.0 * mad)
    model.threshold = threshold
    return threshold, normal_scores


def score_observation(
    model: TexturePatchCore,
    crop: np.ndarray,
    cand: CircleCandidate,
    time_sec: float,
    frame_idx: int,
) -> Tuple[Observation, np.ndarray]:
    patch_score, heat, _ = model.score(crop)
    hybrid_score, yellow, sat, _ = hybrid_from_crop(patch_score, crop)
    threshold = model.threshold if model.threshold is not None else float("inf")
    obs = Observation(
        time_sec=float(time_sec),
        frame_idx=int(frame_idx),
        cx=cand.cx,
        cy=cand.cy,
        r=cand.r,
        detection_score=cand.detection_score,
        yellow_ratio=yellow,
        saturation=sat,
        value=cand.value,
        patchcore_score=float(patch_score),
        hybrid_score=hybrid_score,
        is_defect=bool(hybrid_score > threshold),
    )
    return obs, heat


def scan_video(video_path: Path, cfg: DetectorConfig, model: TexturePatchCore) -> Tuple[List[Observation], dict]:
    cap, fps, frame_count, _, _ = open_video(video_path)
    duration = frame_count / fps if fps else 0.0
    end_sec = cfg.scan_end_sec if cfg.scan_end_sec is not None else duration
    detector = GateCircleDetector(cfg)
    helper = TexturePatchCore(cfg)
    helper.scaler = model.scaler
    helper.nn = model.nn
    helper.memory = model.memory
    helper.threshold = model.threshold
    helper.patch_mask = model.patch_mask
    observations: List[Observation] = []
    heatmaps = {}
    times = list(iter_times(cfg.scan_start_sec, end_sec, cfg.scan_step_sec))
    for t in tqdm(times, desc="scan_video"):
        ok, frame, frame_idx = read_frame_at(cap, t)
        if not ok or frame is None:
            continue
        cand, _ = detector.detect(frame)
        if cand is None:
            continue
        crop = helper.crop_circle(frame, cand)
        obs, heat = score_observation(helper, crop, cand, t, frame_idx)
        observations.append(obs)
        heatmaps[round(t, 4)] = heat
    cap.release()
    return observations, heatmaps


def group_events(observations: Sequence[Observation], cfg: DetectorConfig) -> List[dict]:
    defects = [o for o in observations if o.is_defect]
    events = []
    current: List[Observation] = []
    for obs in defects:
        if current and obs.time_sec - current[-1].time_sec > cfg.defect_min_gap_sec:
            if len(current) >= cfg.min_event_observations:
                events.append(summarize_event(current))
            current = []
        current.append(obs)
    if current and len(current) >= cfg.min_event_observations:
        events.append(summarize_event(current))
    return events


def apply_event_filter(observations: Sequence[Observation], events: Sequence[dict]) -> List[Observation]:
    filtered: List[Observation] = []
    for obs in observations:
        keep = any(float(event["start_sec"]) <= obs.time_sec <= float(event["end_sec"]) for event in events)
        filtered.append(
            Observation(
                time_sec=obs.time_sec,
                frame_idx=obs.frame_idx,
                cx=obs.cx,
                cy=obs.cy,
                r=obs.r,
                detection_score=obs.detection_score,
                yellow_ratio=obs.yellow_ratio,
                saturation=obs.saturation,
                value=obs.value,
                patchcore_score=obs.patchcore_score,
                hybrid_score=obs.hybrid_score,
                is_defect=bool(obs.is_defect and keep),
            )
        )
    return filtered


def summarize_event(group: Sequence[Observation]) -> dict:
    best = max(group, key=lambda o: o.hybrid_score)
    return {
        "start_sec": float(group[0].time_sec),
        "end_sec": float(group[-1].time_sec),
        "duration_sec": float(group[-1].time_sec - group[0].time_sec),
        "observations": len(group),
        "peak_time_sec": float(best.time_sec),
        "peak_hybrid_score": float(best.hybrid_score),
        "peak_patchcore_score": float(best.patchcore_score),
        "peak_saturation": float(best.saturation),
        "peak_yellow_ratio": float(best.yellow_ratio),
        "peak_cx": int(best.cx),
        "peak_cy": int(best.cy),
    }


def normalize_heat(heat: np.ndarray, threshold: float) -> np.ndarray:
    h = heat.astype(np.float32).copy()
    scale = max(float(np.percentile(h[h > 0], 95)) if np.any(h > 0) else threshold, threshold, 1e-6)
    h = np.clip(h / scale, 0, 1)
    return h


def heatmap_overlay(crop_bgr: np.ndarray, heat: np.ndarray, threshold: float, alpha: float = 0.50) -> np.ndarray:
    crop = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    h = normalize_heat(heat, threshold)
    up = cv2.resize(h, (224, 224), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap((up * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(crop, 1 - alpha, color, alpha, 0)


def draw_label(img: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int], scale: float = 0.72) -> None:
    x, y = xy
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 8, y + 6), (12, 15, 22), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def annotate_frame(
    frame_bgr: np.ndarray,
    obs: Optional[Observation],
    crop: Optional[np.ndarray],
    heat: Optional[np.ndarray],
    cfg: DetectorConfig,
    threshold: float,
    recent: Sequence[Observation],
) -> np.ndarray:
    canvas = frame_bgr.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (255, 180, 0), 2)
    cv2.line(overlay, (cfg.gate_x, cfg.roi_y1), (cfg.gate_x, cfg.roi_y2), (255, 255, 0), 2)
    cv2.arrowedLine(overlay, (700, 235), (1120, 235), (255, 255, 0), 3, tipLength=0.03)
    cv2.putText(overlay, "FLOW", (865, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    canvas = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)

    if obs is not None:
        color = (0, 65, 255) if obs.is_defect else (0, 220, 90)
        cv2.circle(canvas, (obs.cx, obs.cy), obs.r, color, 4)
        cv2.circle(canvas, (obs.cx, obs.cy), 3, (255, 255, 255), -1)
        status = "DEFECT" if obs.is_defect else "NORMAL"
        draw_label(canvas, f"{status}  S={obs.hybrid_score:.2f}  T={threshold:.2f}", (obs.cx - 110, obs.cy - obs.r - 14), color)

    # Right diagnostic glass panel.
    x0, y0, x1, y1 = 1418, 42, 1890, 1010
    panel = canvas.copy()
    cv2.rectangle(panel, (x0, y0), (x1, y1), (10, 13, 20), -1)
    canvas = cv2.addWeighted(panel, 0.74, canvas, 0.26, 0)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 210, 255), 2)
    cv2.putText(canvas, "PATCHCORE DIAGNOSTICS", (x0 + 22, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 250, 255), 2)

    if obs is not None and crop is not None and heat is not None:
        crop_small = cv2.resize(crop, (190, 190), interpolation=cv2.INTER_AREA)
        hm_small = cv2.resize(heatmap_overlay(crop, heat, threshold), (190, 190), interpolation=cv2.INTER_AREA)
        canvas[y0 + 70 : y0 + 260, x0 + 24 : x0 + 214] = crop_small
        canvas[y0 + 70 : y0 + 260, x0 + 250 : x0 + 440] = hm_small
        cv2.putText(canvas, "CROP", (x0 + 82, y0 + 285), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 225, 240), 1)
        cv2.putText(canvas, "HEATMAP", (x0 + 302, y0 + 285), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 225, 240), 1)

        draw_metric_bar(canvas, "score", obs.hybrid_score, threshold * 1.7, x0 + 35, y0 + 345, x0 + 430, color=(0, 65, 255) if obs.is_defect else (0, 220, 90), threshold=threshold)
        draw_metric_bar(canvas, "saturation", obs.saturation, 0.70, x0 + 35, y0 + 425, x0 + 430, color=(255, 190, 65), threshold=0.50)
        draw_metric_bar(canvas, "yellow", obs.yellow_ratio, 1.0, x0 + 35, y0 + 505, x0 + 430, color=(60, 230, 255), threshold=0.80)

    draw_timeline_strip(canvas, recent, threshold, (x0 + 35, y0 + 610, x0 + 430, y0 + 900))
    return canvas


def draw_metric_bar(
    img: np.ndarray,
    name: str,
    value: float,
    vmax: float,
    x0: int,
    y0: int,
    x1: int,
    color: Tuple[int, int, int],
    threshold: Optional[float] = None,
) -> None:
    cv2.putText(img, f"{name.upper()}  {value:.3f}", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 240, 245), 1)
    cv2.rectangle(img, (x0, y0), (x1, y0 + 22), (42, 48, 62), -1)
    frac = min(1.0, max(0.0, value / max(vmax, 1e-6)))
    cv2.rectangle(img, (x0, y0), (int(x0 + (x1 - x0) * frac), y0 + 22), color, -1)
    if threshold is not None:
        tx = int(x0 + (x1 - x0) * min(1.0, threshold / max(vmax, 1e-6)))
        cv2.line(img, (tx, y0 - 6), (tx, y0 + 28), (255, 255, 255), 1)


def draw_timeline_strip(img: np.ndarray, observations: Sequence[Observation], threshold: float, box: Tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cv2.putText(img, "RECENT SCORE TRACE", (x0, y0 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 240, 245), 1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (24, 28, 38), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (85, 96, 120), 1)
    if len(observations) < 2:
        return
    window = observations[-80:]
    vals = np.asarray([o.hybrid_score for o in window], dtype=np.float32)
    vmax = max(float(vals.max()), threshold * 1.35, 1e-6)
    th_y = int(y1 - (y1 - y0) * min(1.0, threshold / vmax))
    cv2.line(img, (x0, th_y), (x1, th_y), (255, 255, 255), 1)
    pts = []
    for idx, val in enumerate(vals):
        x = int(x0 + (x1 - x0) * idx / max(1, len(vals) - 1))
        y = int(y1 - (y1 - y0) * min(1.0, float(val) / vmax))
        pts.append((x, y))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(img, p0, p1, (0, 220, 255), 2)
    for idx, obs in enumerate(window):
        if obs.is_defect:
            x = int(x0 + (x1 - x0) * idx / max(1, len(window) - 1))
            cv2.circle(img, (x, pts[idx][1]), 4, (0, 65, 255), -1)


def save_csv(path: Path, rows: Sequence[Observation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(Observation.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_event_csv(path: Path, events: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "start_sec",
        "end_sec",
        "duration_sec",
        "observations",
        "peak_time_sec",
        "peak_hybrid_score",
        "peak_patchcore_score",
        "peak_saturation",
        "peak_yellow_ratio",
        "peak_cx",
        "peak_cy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in events:
            writer.writerow(row)


def pil_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_training_mosaic(crops: Sequence[np.ndarray], out_path: Path) -> None:
    if not crops:
        return
    rng = np.random.default_rng(7)
    idx = rng.choice(len(crops), size=min(64, len(crops)), replace=False)
    tile = 112
    cols = 8
    rows = int(math.ceil(len(idx) / cols))
    header = 76
    img = Image.new("RGB", (cols * tile, rows * tile + header), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    font = pil_font(24)
    draw.text((22, 22), "Normal Memory Bank Samples", fill=(235, 245, 255), font=font)
    for n, i in enumerate(idx):
        crop = cv2.cvtColor(cv2.resize(crops[int(i)], (tile, tile)), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop)
        x = (n % cols) * tile
        y = header + (n // cols) * tile
        img.paste(pil, (x, y))
        ImageDraw.Draw(img).rectangle((x, y, x + tile - 1, y + tile - 1), outline=(80, 210, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_score_timeline(observations: Sequence[Observation], threshold: float, out_path: Path) -> None:
    width, height = 1800, 620
    img = Image.new("RGB", (width, height), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    title_font = pil_font(32)
    small_font = pil_font(18)
    draw.text((36, 28), "Video Anomaly Score Timeline", fill=(235, 245, 255), font=title_font)
    if not observations:
        img.save(out_path)
        return
    xs = np.asarray([o.time_sec for o in observations], dtype=np.float32)
    ys = np.asarray([o.hybrid_score for o in observations], dtype=np.float32)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_max = max(float(ys.max()), threshold * 1.4, 1e-6)
    plot = (80, 110, width - 60, height - 90)
    px0, py0, px1, py1 = plot
    draw.rectangle(plot, outline=(88, 105, 130), width=2)
    for sec, label, fill in [(0, "normal training", (38, 70, 62)), (360, "6 min boundary", (70, 56, 38)), (374, "defect note", (72, 36, 36))]:
        if x_min <= sec <= x_max:
            x = px0 + int((sec - x_min) / max(1e-6, x_max - x_min) * (px1 - px0))
            draw.line((x, py0, x, py1), fill=fill, width=3)
            draw.text((x + 6, py0 + 10), label, fill=(210, 220, 235), font=small_font)
    th_y = py1 - int(min(1.0, threshold / y_max) * (py1 - py0))
    draw.line((px0, th_y, px1, th_y), fill=(245, 245, 245), width=2)
    draw.text((px0 + 8, th_y - 26), f"threshold {threshold:.3f}", fill=(245, 245, 245), font=small_font)

    pts = []
    for xval, yval in zip(xs, ys):
        x = px0 + int((float(xval) - x_min) / max(1e-6, x_max - x_min) * (px1 - px0))
        y = py1 - int(min(1.0, float(yval) / y_max) * (py1 - py0))
        pts.append((x, y))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(0, 210, 255), width=2)
    for obs, p in zip(observations, pts):
        if obs.is_defect:
            draw.ellipse((p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4), fill=(255, 70, 60))
    draw.text((px0, py1 + 28), f"{x_min:.1f}s", fill=(190, 205, 220), font=small_font)
    draw.text((px1 - 80, py1 + 28), f"{x_max:.1f}s", fill=(190, 205, 220), font=small_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_feature_scatter(video_path: Path, cfg: DetectorConfig, train_crops: Sequence[np.ndarray], observations: Sequence[Observation], out_path: Path) -> None:
    def crop_summary(crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (96, 96))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
        yy, xx = np.ogrid[:96, :96]
        mask = (xx - 48) ** 2 + (yy - 48) ** 2 <= 31**2
        pix = hsv[mask]
        hist_h = np.histogram(pix[:, 0], bins=12, range=(0, 180), density=True)[0]
        hist_s = np.histogram(pix[:, 1], bins=8, range=(0, 255), density=True)[0]
        return np.r_[hist_h, hist_s, pix[:, 1].mean() / 255.0, pix[:, 2].mean() / 255.0, gray[mask].std() / 255.0]

    sample_train = list(train_crops)[:: max(1, len(train_crops) // 180)]
    if len(sample_train) < 3 or not observations:
        return
    train_feats = np.vstack([crop_summary(c) for c in sample_train])
    cap, _, _, _, _ = open_video(video_path)
    helper = TexturePatchCore(cfg)
    obs_crops = []
    for obs in observations:
        ok, frame, _ = read_frame_at(cap, obs.time_sec)
        if not ok or frame is None:
            continue
        cand = CircleCandidate(obs.cx, obs.cy, obs.r, obs.detection_score, obs.yellow_ratio, obs.saturation, obs.value)
        obs_crops.append((obs, helper.crop_circle(frame, cand)))
    cap.release()
    if not obs_crops:
        return
    obs_feats = np.vstack([crop_summary(crop) for _, crop in obs_crops])
    features = np.vstack([train_feats, obs_feats])
    coords = PCA(n_components=2, random_state=7).fit_transform(StandardScaler().fit_transform(features))
    width, height = 1200, 860
    img = Image.new("RGB", (width, height), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    title_font = pil_font(30)
    small_font = pil_font(16)
    draw.text((34, 28), "Feature Space Overview", fill=(235, 245, 255), font=title_font)
    plot = (70, 100, width - 60, height - 70)
    draw.rectangle(plot, outline=(88, 105, 130), width=2)
    c = coords
    minxy = c.min(axis=0)
    maxxy = c.max(axis=0)
    span = np.maximum(maxxy - minxy, 1e-6)

    def project(pt):
        x = plot[0] + int((pt[0] - minxy[0]) / span[0] * (plot[2] - plot[0]))
        y = plot[3] - int((pt[1] - minxy[1]) / span[1] * (plot[3] - plot[1]))
        return x, y

    for pt in coords[: len(sample_train)]:
        x, y = project(pt)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(80, 220, 120))
    for (obs, _), pt in zip(obs_crops, coords[len(sample_train) :]):
        x, y = project(pt)
        color = (255, 70, 60) if obs.is_defect else (0, 210, 255)
        r = 5 if obs.is_defect else 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    draw.text((80, height - 48), "green: normal memory   cyan: scanned normal   red: defect candidates", fill=(205, 215, 230), font=small_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_defect_gallery(
    video_path: Path,
    cfg: DetectorConfig,
    model: TexturePatchCore,
    observations: Sequence[Observation],
    heatmaps: dict,
    out_path: Path,
) -> None:
    final_defects = [o for o in observations if o.is_defect]
    top = sorted(final_defects or observations, key=lambda o: o.hybrid_score, reverse=True)[:16]
    if not top:
        return
    cap, _, _, _, _ = open_video(video_path)
    helper = TexturePatchCore(cfg)
    helper.scaler = model.scaler
    helper.nn = model.nn
    helper.memory = model.memory
    helper.threshold = model.threshold
    helper.patch_mask = model.patch_mask
    tile_w, tile_h = 300, 370
    cols = 4
    rows = int(math.ceil(len(top) / cols))
    img = Image.new("RGB", (cols * tile_w, rows * tile_h + 78), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    title_font = pil_font(28)
    small_font = pil_font(16)
    draw.text((28, 22), "Top Defect Candidates", fill=(235, 245, 255), font=title_font)
    for idx, obs in enumerate(top):
        ok, frame, _ = read_frame_at(cap, obs.time_sec)
        if not ok or frame is None:
            continue
        cand = CircleCandidate(obs.cx, obs.cy, obs.r, obs.detection_score, obs.yellow_ratio, obs.saturation, obs.value)
        crop = helper.crop_circle(frame, cand)
        heat = heatmaps.get(round(obs.time_sec, 4))
        if heat is None:
            _, heat, _ = helper.score(crop)
        overlay = heatmap_overlay(crop, heat, model.threshold or 1.0)
        pair = np.hstack([cv2.resize(crop, (128, 128)), cv2.resize(overlay, (128, 128))])
        pair_rgb = cv2.cvtColor(pair, cv2.COLOR_BGR2RGB)
        x = (idx % cols) * tile_w + 22
        y = 78 + (idx // cols) * tile_h + 18
        img.paste(Image.fromarray(pair_rgb), (x, y))
        draw.rectangle((x, y, x + 256, y + 128), outline=(255, 70, 60), width=2)
        draw.text((x, y + 146), f"t={obs.time_sec:.2f}s  score={obs.hybrid_score:.3f}", fill=(245, 245, 245), font=small_font)
        draw.text((x, y + 170), f"patch={obs.patchcore_score:.3f}  sat={obs.saturation:.3f}", fill=(205, 215, 230), font=small_font)
        draw.text((x, y + 194), f"center=({obs.cx},{obs.cy}) r={obs.r}", fill=(205, 215, 230), font=small_font)
    cap.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_roi_calibration(video_path: Path, cfg: DetectorConfig, out_path: Path, time_sec: float = 374.0) -> None:
    cap, _, _, _, _ = open_video(video_path)
    ok, frame, _ = read_frame_at(cap, time_sec)
    cap.release()
    if not ok or frame is None:
        return
    detector = GateCircleDetector(cfg)
    best, candidates = detector.detect(frame)
    vis = frame.copy()
    cv2.rectangle(vis, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (255, 180, 0), 3)
    cv2.line(vis, (cfg.gate_x, cfg.roi_y1), (cfg.gate_x, cfg.roi_y2), (255, 255, 0), 3)
    for cand in candidates[:10]:
        color = (0, 255, 0) if best is cand else (90, 90, 255)
        cv2.circle(vis, (cand.cx, cand.cy), cand.r, color, 3)
        cv2.putText(vis, f"{cand.detection_score:.2f}", (cand.cx - 36, cand.cy - cand.r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    draw_label(vis, "virtual inspection gate", (cfg.gate_x + 14, cfg.roi_y1 + 34), (255, 255, 0), 0.72)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def save_annotated_preview(
    video_path: Path,
    cfg: DetectorConfig,
    model: TexturePatchCore,
    out_path: Path,
) -> None:
    cap, fps, _, width, height = open_video(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    detector = GateCircleDetector(cfg)
    helper = TexturePatchCore(cfg)
    helper.scaler = model.scaler
    helper.nn = model.nn
    helper.memory = model.memory
    helper.threshold = model.threshold
    helper.patch_mask = model.patch_mask
    recent: List[Observation] = []
    times = list(iter_times(cfg.preview_start_sec, cfg.preview_end_sec, cfg.preview_step_sec))
    for t in tqdm(times, desc="preview"):
        ok, frame, frame_idx = read_frame_at(cap, t)
        if not ok or frame is None:
            continue
        cand, _ = detector.detect(frame)
        obs = None
        crop = None
        heat = None
        if cand is not None:
            crop = helper.crop_circle(frame, cand)
            obs, heat = score_observation(helper, crop, cand, t, frame_idx)
            recent.append(obs)
        annotated = annotate_frame(frame, obs, crop, heat, cfg, model.threshold or 1.0, recent)
        writer.write(annotated)
    writer.release()
    cap.release()


def load_model_bundle(model_dir: Path) -> Tuple[TexturePatchCore, DetectorConfig, List[float]]:
    with (model_dir / "texture_patchcore_model.pkl").open("rb") as f:
        bundle = pickle.load(f)
    cfg_fields = set(DetectorConfig.__dataclass_fields__.keys())
    cfg_data = {k: v for k, v in bundle["config"].items() if k in cfg_fields}
    cfg = DetectorConfig(**cfg_data)
    model = TexturePatchCore(cfg)
    model.scaler = bundle["scaler"]
    model.memory = bundle["memory"]
    model.threshold = float(bundle["threshold"])
    model.patch_mask = bundle["patch_mask"]
    model.nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    model.nn.fit(model.memory)
    return model, cfg, list(bundle.get("threshold_scores", []))


def load_observations_csv(path: Path) -> List[Observation]:
    rows: List[Observation] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                Observation(
                    time_sec=float(row["time_sec"]),
                    frame_idx=int(row["frame_idx"]),
                    cx=int(row["cx"]),
                    cy=int(row["cy"]),
                    r=int(row["r"]),
                    detection_score=float(row["detection_score"]),
                    yellow_ratio=float(row["yellow_ratio"]),
                    saturation=float(row["saturation"]),
                    value=float(row["value"]),
                    patchcore_score=float(row["patchcore_score"]),
                    hybrid_score=float(row["hybrid_score"]),
                    is_defect=row["is_defect"].strip().lower() == "true",
                )
            )
    return rows


def review_state(obs: Optional[Observation], threshold: float) -> Tuple[str, Tuple[int, int, int]]:
    if obs is None:
        return "NO TARGET", (170, 180, 190)
    if obs.is_defect:
        return "DEFECT", (0, 65, 255)
    if obs.hybrid_score > threshold:
        return "SPIKE / FILTERED", (0, 185, 255)
    return "NORMAL", (0, 220, 90)


def annotate_review_frame(
    frame_bgr: np.ndarray,
    obs: Optional[Observation],
    crop: Optional[np.ndarray],
    heat: Optional[np.ndarray],
    cfg: DetectorConfig,
    threshold: float,
    recent: Sequence[Observation],
    time_sec: float,
    sample_step_sec: float,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    state, color = review_state(obs, threshold)

    veil = canvas.copy()
    cv2.rectangle(veil, (0, 0), (1920, 94), (8, 11, 18), -1)
    cv2.rectangle(veil, (cfg.roi_x1, cfg.roi_y1), (cfg.roi_x2, cfg.roi_y2), (255, 185, 0), 3)
    cv2.line(veil, (cfg.gate_x, cfg.roi_y1), (cfg.gate_x, cfg.roi_y2), (255, 255, 0), 3)
    cv2.arrowedLine(veil, (cfg.roi_x1 + 36, cfg.roi_y1 - 46), (cfg.roi_x2 - 36, cfg.roi_y1 - 46), (255, 255, 0), 3, tipLength=0.03)
    canvas = cv2.addWeighted(veil, 0.82, canvas, 0.18, 0)

    cv2.putText(canvas, "2FPS SAMPLED REVIEW VIDEO", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (240, 248, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"t={time_sec:06.2f}s   sample={sample_step_sec:.2f}s   threshold={threshold:.3f}   event rule: >=2 consecutive samples",
        (28, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (210, 225, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "ROI + VIRTUAL GATE", (cfg.roi_x1 + 18, cfg.roi_y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2, cv2.LINE_AA)

    if obs is not None:
        cv2.circle(canvas, (obs.cx, obs.cy), obs.r, color, 4)
        cv2.circle(canvas, (obs.cx, obs.cy), 4, (255, 255, 255), -1)
        compare = ">" if obs.hybrid_score > threshold else "<="
        label = f"{state}  score={obs.hybrid_score:.3f} {compare} T={threshold:.3f}"
        draw_label(canvas, label, (max(18, obs.cx - 182), max(118, obs.cy - obs.r - 18)), color, scale=0.78)
        cv2.putText(
            canvas,
            f"patch={obs.patchcore_score:.3f}  yellow={obs.yellow_ratio:.3f}  sat={obs.saturation:.3f}",
            (max(18, obs.cx - 182), min(frame_bgr.shape[0] - 28, obs.cy + obs.r + 34)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    else:
        draw_label(canvas, "NO TARGET DETECTED AT THIS SAMPLE", (cfg.roi_x1 + 42, cfg.roi_y1 + 56), color, scale=0.78)

    # Diagnostic panel with threshold, crop and PatchCore heatmap.
    x0, y0, x1, y1 = 1426, 116, 1890, 1018
    panel = canvas.copy()
    cv2.rectangle(panel, (x0, y0), (x1, y1), (10, 13, 20), -1)
    canvas = cv2.addWeighted(panel, 0.76, canvas, 0.24, 0)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
    cv2.putText(canvas, "FRAME DIAGNOSTICS", (x0 + 24, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 250, 255), 2)

    if obs is not None and crop is not None and heat is not None:
        crop_small = cv2.resize(crop, (190, 190), interpolation=cv2.INTER_AREA)
        hm_small = cv2.resize(heatmap_overlay(crop, heat, threshold), (190, 190), interpolation=cv2.INTER_AREA)
        canvas[y0 + 62 : y0 + 252, x0 + 24 : x0 + 214] = crop_small
        canvas[y0 + 62 : y0 + 252, x0 + 250 : x0 + 440] = hm_small
        cv2.putText(canvas, "SELECTED CROP", (x0 + 44, y0 + 278), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 225, 240), 1)
        cv2.putText(canvas, "PATCH HEATMAP", (x0 + 278, y0 + 278), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 225, 240), 1)
        draw_metric_bar(canvas, "score", obs.hybrid_score, threshold * 1.6, x0 + 34, y0 + 340, x0 + 430, color=color, threshold=threshold)
        draw_metric_bar(canvas, "patchcore", obs.patchcore_score, threshold * 1.4, x0 + 34, y0 + 416, x0 + 430, color=(0, 210, 255), threshold=threshold)
        draw_metric_bar(canvas, "yellow", obs.yellow_ratio, 1.0, x0 + 34, y0 + 492, x0 + 430, color=(55, 225, 255), threshold=0.80)
        draw_metric_bar(canvas, "saturation", obs.saturation, 0.70, x0 + 34, y0 + 568, x0 + 430, color=(255, 190, 65), threshold=0.50)
    else:
        cv2.putText(canvas, "No crop / heatmap for this frame", (x0 + 34, y0 + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (210, 225, 240), 2)

    draw_timeline_strip(canvas, recent, threshold, (x0 + 34, y0 + 676, x0 + 430, y0 + 866))
    return canvas


def save_sampled_review_video(
    video_path: Path,
    observations_path: Path,
    model_dir: Path,
    out_path: Path,
    fps_out: float = 2.0,
    sample_step_sec: float = 0.5,
    snapshots_dir: Optional[Path] = None,
) -> None:
    model, cfg, _ = load_model_bundle(model_dir)
    observations = load_observations_csv(observations_path)
    obs_by_time = {round(obs.time_sec, 2): obs for obs in observations}
    if observations:
        start_sec = min(obs.time_sec for obs in observations)
        end_sec = max(obs.time_sec for obs in observations)
    else:
        start_sec = cfg.scan_start_sec
        cap_tmp, fps, frame_count, _, _ = open_video(video_path)
        end_sec = frame_count / fps
        cap_tmp.release()

    cap, _, _, width, height = open_video(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (width, height))
    helper = TexturePatchCore(cfg)
    helper.scaler = model.scaler
    helper.nn = model.nn
    helper.memory = model.memory
    helper.threshold = model.threshold
    helper.patch_mask = model.patch_mask
    recent: List[Observation] = []
    snapshot_targets = []
    if snapshots_dir is not None:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_targets = [start_sec, 374.0, 397.0, end_sec]

    for t in tqdm(list(iter_times(start_sec, end_sec, sample_step_sec)), desc="sampled_review_video"):
        ok, frame, _ = read_frame_at(cap, t)
        if not ok or frame is None:
            continue
        obs = obs_by_time.get(round(t, 2))
        crop = None
        heat = None
        if obs is not None:
            cand = CircleCandidate(obs.cx, obs.cy, obs.r, obs.detection_score, obs.yellow_ratio, obs.saturation, obs.value)
            crop = helper.crop_circle(frame, cand)
            _, heat, _ = helper.score(crop)
            recent.append(obs)
        annotated = annotate_review_frame(frame, obs, crop, heat, cfg, model.threshold or 1.0, recent, t, sample_step_sec)
        writer.write(annotated)
        for target in list(snapshot_targets):
            if abs(t - target) <= sample_step_sec / 2:
                state, _ = review_state(obs, model.threshold or 1.0)
                safe_state = state.lower().replace(" / ", "_").replace(" ", "_")
                cv2.imwrite(str(snapshots_dir / f"review_{t:07.2f}s_{safe_state}.jpg"), annotated)
                snapshot_targets.remove(target)

    writer.release()
    cap.release()


def save_model_bundle(model: TexturePatchCore, cfg: DetectorConfig, out_dir: Path, threshold_scores: Sequence[float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "texture_patchcore_model.pkl").open("wb") as f:
        pickle.dump(
            {
                "config": asdict(cfg),
                "scaler": model.scaler,
                "memory": model.memory,
                "threshold": model.threshold,
                "patch_mask": model.patch_mask,
                "threshold_scores": list(map(float, threshold_scores)),
            },
            f,
        )
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "threshold": model.threshold}, f, indent=2)


def run_pipeline(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    out_dir = Path(args.out)
    cfg = DetectorConfig(
        train_end_sec=args.train_end,
        val_start_sec=args.val_start,
        val_end_sec=args.val_end,
        scan_step_sec=args.scan_step,
        preview_start_sec=args.preview_start,
        preview_end_sec=args.preview_end,
        max_memory_patches=args.max_memory,
    )
    if args.scan_start is not None:
        cfg.scan_start_sec = args.scan_start
    if args.scan_end is not None:
        cfg.scan_end_sec = args.scan_end

    train_crops, train_meta = collect_crops(video_path, cfg, cfg.train_start_sec, cfg.train_end_sec, cfg.train_step_sec, "collect_train_normal")
    val_crops, val_meta = collect_crops(video_path, cfg, cfg.val_start_sec, cfg.val_end_sec, cfg.train_step_sec, "collect_val_normal")
    if len(train_crops) < 20:
        raise RuntimeError(f"Too few normal crops were collected: {len(train_crops)}")

    model = TexturePatchCore(cfg)
    model.fit(train_crops)
    threshold, threshold_scores = set_threshold(model, train_crops, val_crops, cfg)
    print(f"normal crops: {len(train_crops)}  validation crops: {len(val_crops)}")
    print(f"memory patches: {model.memory.shape[0] if model.memory is not None else 0}")
    print(f"threshold: {threshold:.6f}")

    observations, heatmaps = scan_video(video_path, cfg, model)
    raw_events = group_events(observations, cfg)
    observations = apply_event_filter(observations, raw_events)
    events = group_events(observations, cfg)
    print(f"observations: {len(observations)}  defect events: {len(events)}")

    save_csv(out_dir / "observations.csv", observations)
    save_event_csv(out_dir / "defect_events.csv", events)
    save_model_bundle(model, cfg, out_dir / "model", threshold_scores)
    save_roi_calibration(video_path, cfg, out_dir / "visuals" / "01_roi_calibration.jpg")
    save_training_mosaic(train_crops, out_dir / "visuals" / "02_normal_memory_mosaic.jpg")
    save_score_timeline(observations, threshold, out_dir / "visuals" / "03_score_timeline.png")
    save_defect_gallery(video_path, cfg, model, observations, heatmaps, out_dir / "visuals" / "04_defect_gallery.jpg")
    save_feature_scatter(video_path, cfg, train_crops, observations, out_dir / "visuals" / "05_feature_space.png")
    save_annotated_preview(video_path, cfg, model, out_dir / "visuals" / "06_annotated_diagnostic_preview.mp4")

    summary = {
        "video": str(video_path),
        "normal_crops": len(train_crops),
        "validation_crops": len(val_crops),
        "observations": len(observations),
        "defect_events": events,
        "threshold": threshold,
        "outputs": {
            "observations_csv": str(out_dir / "observations.csv"),
            "defect_events_csv": str(out_dir / "defect_events.csv"),
            "visuals_dir": str(out_dir / "visuals"),
            "model_dir": str(out_dir / "model"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_review_video(args: argparse.Namespace) -> None:
    save_sampled_review_video(
        video_path=Path(args.video),
        observations_path=Path(args.observations),
        model_dir=Path(args.model_dir),
        out_path=Path(args.out),
        fps_out=args.fps,
        sample_step_sec=args.sample_step,
        snapshots_dir=Path(args.snapshots_dir) if args.snapshots_dir else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Self-supervised gate PatchCore defect detector for conveyor video.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Train on the normal video segment, scan the video, and create diagnostics.")
    run.add_argument("--video", default="20241031_222226.mp4")
    run.add_argument("--out", default="defect_outputs")
    run.add_argument("--train-end", type=float, default=350.0)
    run.add_argument("--val-start", type=float, default=350.0)
    run.add_argument("--val-end", type=float, default=360.0)
    run.add_argument("--scan-start", type=float, default=None)
    run.add_argument("--scan-end", type=float, default=None)
    run.add_argument("--scan-step", type=float, default=0.25)
    run.add_argument("--preview-start", type=float, default=360.0)
    run.add_argument("--preview-end", type=float, default=430.0)
    run.add_argument("--max-memory", type=int, default=4096)
    run.set_defaults(func=run_pipeline)

    review = sub.add_parser("review-video", help="Create a 2fps sampled review video from saved observations and model.")
    review.add_argument("--video", default="20241031_222226.mp4")
    review.add_argument("--observations", default="defect_outputs/observations.csv")
    review.add_argument("--model-dir", default="defect_outputs/model")
    review.add_argument("--out", default="defect_outputs/visuals/07_sampled_2fps_review.mp4")
    review.add_argument("--fps", type=float, default=2.0)
    review.add_argument("--sample-step", type=float, default=0.5)
    review.add_argument("--snapshots-dir", default="defect_outputs/visuals/07_review_snapshots")
    review.set_defaults(func=run_review_video)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
