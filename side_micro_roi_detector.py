from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import cap_defect_detection_pipeline as caploc


Color = Tuple[int, int, int]


@dataclass
class MicroConfig:
    right_half_x: int = 960
    upper_y1: int = 350
    upper_y2: int = 495
    lower_y1: int = 835
    lower_y2: int = 960
    win_w: int = 84
    win_h: int = 68
    scan_step: int = 18
    candidate_dx_left: int = -92
    candidate_dx_near_left: int = -64
    candidate_dx_right: int = 192
    min_candidate_x: int = 1000
    sample_step_sec: float = 0.5
    review_fps: float = 2.0
    present_threshold: float = 0.48
    missing_threshold: float = 0.30
    background_threshold: float = 0.56
    max_candidates_per_band: int = 4
    nms_iou: float = 0.18
    replacement_missing_threshold: float = 0.47
    min_event_observations: int = 2
    event_gap_sec: float = 1.25
    random_seed: int = 17


@dataclass
class MicroDetection:
    time_sec: float
    frame_idx: int
    band: str
    x: int
    y: int
    w: int
    h: int
    state: str
    final_state: str
    p_present: float
    p_missing: float
    p_background: float
    score: float
    white_ratio: float = 0.0
    sleeve_blob: float = 0.0
    edge_density: float = 0.0
    rule: str = ""
    event_id: int = 0


def open_video(path: Path) -> Tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return cap, fps, frame_count, width, height


def read_frame_at(cap: cv2.VideoCapture, fps: float, time_sec: float) -> Tuple[bool, np.ndarray | None, int]:
    frame_idx = max(0, int(round(time_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return ok, frame, frame_idx


def iter_times(start: float, end: float, step: float) -> Iterable[float]:
    t = start
    eps = step / 10.0
    while t <= end + eps:
        yield round(t, 4)
        t += step


def pil_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ["arial.ttf", "msyh.ttc", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union else 0.0


def crop_features(crop_bgr: np.ndarray, size: Tuple[int, int] = (96, 76)) -> np.ndarray:
    crop = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    features: List[float] = []

    for img in [hsv, lab]:
        for channel in cv2.split(img):
            features.extend([float(channel.mean()), float(channel.std()), float(np.percentile(channel, 10)), float(np.percentile(channel, 90))])

    h, s, v = cv2.split(hsv)
    white = ((s < 80) & (v > 115)).astype(np.uint8)
    dark = (v < 85).astype(np.uint8)
    blue = ((h >= 92) & (h <= 132) & (s > 50) & (v > 35)).astype(np.uint8)
    features.extend([float(white.mean()), float(dark.mean()), float(blue.mean())])

    # Spatially pooled texture/color keeps enough shape information for the
    # small side-port patches without needing a heavy model.
    gh, gw = 4, 4
    cell_h, cell_w = gray.shape[0] // gh, gray.shape[1] // gw
    for yy in range(gh):
        for xx in range(gw):
            y1, x1 = yy * cell_h, xx * cell_w
            y2 = gray.shape[0] if yy == gh - 1 else (yy + 1) * cell_h
            x2 = gray.shape[1] if xx == gw - 1 else (xx + 1) * cell_w
            gcell = gray[y1:y2, x1:x2]
            scell = s[y1:y2, x1:x2]
            vcell = v[y1:y2, x1:x2]
            wcell = white[y1:y2, x1:x2]
            features.extend([float(gcell.mean()), float(gcell.std()), float(scell.mean()), float(vcell.mean()), float(wcell.mean())])

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edges = cv2.Canny(gray, 70, 160)
    features.extend([float(mag.mean()), float(mag.std()), float(lap.var()), float(edges.mean() / 255.0)])

    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist = hist / max(1.0, float(hist.sum()))
    features.extend([float(v) for v in hist])
    return np.asarray(features, dtype=np.float32)


def augment_labeled_crop(crop_bgr: np.ndarray, label: str) -> List[np.ndarray]:
    variants = [crop_bgr]
    variants.append(cv2.flip(crop_bgr, 1))
    for alpha, beta in [(0.88, -6), (1.12, 6)]:
        variants.append(cv2.convertScaleAbs(crop_bgr, alpha=alpha, beta=beta))
    if label == "MISSING":
        kernel = np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        variants.append(cv2.filter2D(crop_bgr, -1, kernel))
        variants.append(cv2.GaussianBlur(crop_bgr, (3, 3), 0))
        h, w = crop_bgr.shape[:2]
        for dx, dy in [(-4, 0), (4, 0), (0, -3), (0, 3)]:
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(crop_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            variants.append(shifted)
    return variants


def load_tasks_and_annotations(annotation_dir: Path) -> Tuple[dict, dict]:
    tasks_path = annotation_dir / "tasks.json"
    annotations_path = annotation_dir / "annotations.json"
    if not tasks_path.exists() or not annotations_path.exists():
        raise FileNotFoundError(f"Need tasks.json and annotations.json under {annotation_dir}")
    return json.loads(tasks_path.read_text(encoding="utf-8")), json.loads(annotations_path.read_text(encoding="utf-8"))


def build_training_set(annotation_dir: Path, cfg: MicroConfig, out_dir: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    tasks, annotations = load_tasks_and_annotations(annotation_dir)
    frames = {frame["id"]: frame for frame in tasks["frames"]}
    samples: List[np.ndarray] = []
    labels: List[str] = []
    crop_manifest: List[dict] = []
    crop_dir = out_dir / "training_crops"
    if crop_dir.exists():
        shutil.rmtree(crop_dir)
    for label in ["PRESENT", "MISSING", "BACKGROUND"]:
        (crop_dir / label).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_seed)
    annotated_boxes_by_frame: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for frame_id, anno in annotations["frames"].items():
        frame = frames.get(frame_id)
        if not frame:
            continue
        image = cv2.imread(str(annotation_dir / frame["filename"]))
        if image is None:
            continue
        offset_x = int(frame["offset_x"])
        boxes: List[Tuple[int, int, int, int]] = []
        for idx, obj in enumerate(anno.get("objects", []), start=1):
            if obj.get("type") != "sample" or obj.get("state") not in {"PRESENT", "MISSING"}:
                continue
            x = int(obj["x"] - offset_x)
            y = int(obj["y"])
            w = int(obj["w"])
            h = int(obj["h"])
            x = max(0, min(image.shape[1] - 1, x))
            y = max(0, min(image.shape[0] - 1, y))
            crop = image[y : min(image.shape[0], y + h), x : min(image.shape[1], x + w)]
            if crop.size == 0:
                continue
            label = obj["state"]
            for variant in augment_labeled_crop(crop, label):
                samples.append(crop_features(variant, (cfg.win_w, cfg.win_h)))
                labels.append(label)
            boxes.append((x, y, w, h))
            crop_name = f"{frame_id}_{idx:02d}_{label}.jpg"
            cv2.imwrite(str(crop_dir / label / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
            crop_manifest.append({"frame_id": frame_id, "label": label, "x": obj["x"], "y": y, "w": w, "h": h, "crop": str((crop_dir / label / crop_name).as_posix())})
        annotated_boxes_by_frame[frame_id] = boxes

        background_needed = max(10, len(boxes) * 2)
        bands = [("upper", cfg.upper_y1, cfg.upper_y2), ("lower", cfg.lower_y1, cfg.lower_y2)]
        made = 0
        attempts = 0
        while made < background_needed and attempts < background_needed * 80:
            attempts += 1
            _, y1, y2 = bands[int(rng.integers(0, len(bands)))]
            x = int(rng.integers(0, max(1, image.shape[1] - cfg.win_w)))
            y = int(rng.integers(max(0, y1), max(1, min(image.shape[0] - cfg.win_h, y2 - cfg.win_h))))
            box = (x, y, cfg.win_w, cfg.win_h)
            if any(box_iou(box, existing) > 0.08 for existing in boxes):
                continue
            crop = image[y : y + cfg.win_h, x : x + cfg.win_w]
            if crop.size == 0:
                continue
            samples.append(crop_features(crop, (cfg.win_w, cfg.win_h)))
            labels.append("BACKGROUND")
            crop_name = f"{frame_id}_bg_{made:02d}.jpg"
            cv2.imwrite(str(crop_dir / "BACKGROUND" / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crop_manifest.append({"frame_id": frame_id, "label": "BACKGROUND", "x": x + offset_x, "y": y, "w": cfg.win_w, "h": cfg.win_h, "crop": str((crop_dir / "BACKGROUND" / crop_name).as_posix())})
            made += 1

    if not samples:
        raise RuntimeError("No labeled side micro ROI crops found.")
    manifest_path = out_dir / "training_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(crop_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(crop_manifest)
    return np.vstack(samples), np.asarray(labels), {"manifest": str(manifest_path), "crop_dir": str(crop_dir)}


def train_model(annotation_dir: Path, cfg: MicroConfig, out_dir: Path) -> Tuple[ExtraTreesClassifier, dict]:
    X, y, data_info = build_training_set(annotation_dir, cfg, out_dir)
    model = ExtraTreesClassifier(
        n_estimators=160,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=cfg.random_seed,
        n_jobs=-1,
    )
    report: dict = {"label_counts": {label: int(np.sum(y == label)) for label in sorted(set(y))}, **data_info}
    if len(set(y)) >= 3 and min(np.sum(y == label) for label in set(y)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=cfg.random_seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        report["holdout_report"] = classification_report(y_test, pred, output_dict=True, zero_division=0)
        report["holdout_confusion_matrix"] = {
            "labels": list(model.classes_),
            "matrix": confusion_matrix(y_test, pred, labels=model.classes_).tolist(),
        }
    model.fit(X, y)
    return model, report


def predict_proba_map(model: ExtraTreesClassifier, features: np.ndarray) -> Dict[str, float]:
    probs = model.predict_proba(features.reshape(1, -1))[0]
    mapping = {label: float(prob) for label, prob in zip(model.classes_, probs)}
    for label in ["PRESENT", "MISSING", "BACKGROUND"]:
        mapping.setdefault(label, 0.0)
    return mapping


def classify_probs(probs: Dict[str, float], cfg: MicroConfig) -> Tuple[str, float]:
    p_present = probs["PRESENT"]
    p_missing = probs["MISSING"]
    p_bg = probs["BACKGROUND"]
    objectness = max(p_present, p_missing) - 0.35 * p_bg
    if p_bg >= cfg.background_threshold and max(p_present, p_missing) < 0.62:
        return "BACKGROUND", objectness
    if p_missing >= cfg.missing_threshold and p_missing >= p_present + 0.06:
        return "MISSING", objectness + 0.35 * p_missing
    if p_missing >= cfg.missing_threshold and p_present <= 0.66 and p_missing >= p_bg + 0.12:
        return "MISSING", objectness + 0.35 * p_missing
    if p_present >= cfg.present_threshold and p_present >= p_missing:
        return "PRESENT", objectness + 0.20 * p_present
    return "UNKNOWN", objectness


def sleeve_evidence(crop_bgr: np.ndarray, cfg: MicroConfig) -> Dict[str, float]:
    """Hand-check the tiny opaque sleeve instead of the whole transparent port."""
    crop = cv2.resize(crop_bgr, (cfg.win_w, cfg.win_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, saturation, value = cv2.split(hsv)
    edges = cv2.Canny(gray, 70, 160)

    h, w = gray.shape[:2]
    y1, y2 = int(h * 0.12), int(h * 0.88)
    x1, x2 = int(w * 0.05), int(w * 0.78)
    low_sat_bright = ((saturation < 70) & (value > 115)).astype(np.uint8)
    roi = low_sat_bright[y1:y2, x1:x2]
    edge_roi = edges[y1:y2, x1:x2]

    num, _, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.asarray([], dtype=np.int32)
    sleeve_blob = float(areas.max() / max(1, roi.size)) if areas.size else 0.0
    return {
        "white_ratio": float(roi.mean()) if roi.size else 0.0,
        "sleeve_blob": sleeve_blob,
        "edge_density": float((edge_roi > 0).mean()) if edge_roi.size else 0.0,
    }


def refine_state_with_sleeve_rule(
    probs: Dict[str, float],
    state: str,
    score: float,
    evidence: Dict[str, float],
) -> Tuple[str, float, str]:
    white_ratio = evidence["white_ratio"]
    sleeve_blob = evidence["sleeve_blob"]
    edge_density = evidence["edge_density"]

    solid_sleeve = (
        white_ratio >= 0.40
        and sleeve_blob >= 0.42
        and edge_density <= 0.165
        and probs["MISSING"] < 0.25
    )
    transparent_or_bare_port = (
        edge_density >= 0.185
        and sleeve_blob <= 0.72
        and white_ratio <= 0.78
    ) or (
        edge_density >= 0.09
        and sleeve_blob <= 0.18
        and white_ratio <= 0.34
    )

    if solid_sleeve:
        sleeve_score = 1.05 + sleeve_blob - edge_density + 0.15 * probs["PRESENT"]
        return "PRESENT", max(score, sleeve_score), "solid_sleeve"
    if transparent_or_bare_port:
        missing_score = 0.92 + edge_density + 0.35 * probs["MISSING"] - 0.10 * sleeve_blob
        return "MISSING", max(score, missing_score), "transparent_or_bare_port"
    return state, score, "model"


def scan_frame(frame_bgr: np.ndarray, model: ExtraTreesClassifier, cfg: MicroConfig, time_sec: float, frame_idx: int) -> List[MicroDetection]:
    bands = [("upper", cfg.upper_y1, cfg.upper_y2), ("lower", cfg.lower_y1, cfg.lower_y2)]
    width = frame_bgr.shape[1]
    loc_cfg = caploc.CapDetectorConfig(right_half_x=cfg.right_half_x)
    pairs, _ = caploc.pair_anchors(caploc.detect_blue_anchors(frame_bgr, loc_cfg), loc_cfg)
    x_centers: List[float] = []
    for top_anchor, bottom_anchor in pairs:
        pair_cx = float((top_anchor.cx + bottom_anchor.cx) / 2.0)
        x_centers.extend([pair_cx + cfg.candidate_dx_left, pair_cx + cfg.candidate_dx_near_left, pair_cx + cfg.candidate_dx_right])
    if not x_centers:
        x_centers = [float(x + cfg.win_w / 2.0) for x in range(cfg.right_half_x, max(cfg.right_half_x + 1, width - cfg.win_w + 1), 72)]
    feature_rows: List[np.ndarray] = []
    specs: List[Tuple[str, int, int]] = []
    for band, y1, y2 in bands:
        y_center = int(round((y1 + y2 - cfg.win_h) / 2))
        y = max(0, min(frame_bgr.shape[0] - cfg.win_h, y_center))
        for x_center in x_centers:
            x = int(round(x_center - cfg.win_w / 2.0))
            if x < max(cfg.right_half_x, cfg.min_candidate_x) or x + cfg.win_w > width:
                continue
            crop = frame_bgr[y : y + cfg.win_h, x : x + cfg.win_w]
            if crop.shape[0] != cfg.win_h or crop.shape[1] != cfg.win_w:
                continue
            feature_rows.append(crop_features(crop, (cfg.win_w, cfg.win_h)))
            specs.append((band, x, y))
    if not feature_rows:
        return []

    prob_rows = model.predict_proba(np.vstack(feature_rows))
    by_band: Dict[str, List[MicroDetection]] = {"upper": [], "lower": []}
    for (band, x, y), probs_arr in zip(specs, prob_rows):
        probs = {label: float(prob) for label, prob in zip(model.classes_, probs_arr)}
        for label in ["PRESENT", "MISSING", "BACKGROUND"]:
            probs.setdefault(label, 0.0)
        crop = frame_bgr[y : y + cfg.win_h, x : x + cfg.win_w]
        evidence = sleeve_evidence(crop, cfg)
        state, score = classify_probs(probs, cfg)
        state, score, rule = refine_state_with_sleeve_rule(probs, state, score, evidence)
        if state == "BACKGROUND" and score < 0.30:
            continue
        if state in {"PRESENT", "MISSING", "UNKNOWN"} and score >= 0.34:
            by_band.setdefault(band, []).append(
                MicroDetection(
                    time_sec=float(time_sec),
                    frame_idx=int(frame_idx),
                    band=band,
                    x=int(x),
                    y=int(y),
                    w=cfg.win_w,
                    h=cfg.win_h,
                    state=state,
                    final_state=state,
                    p_present=probs["PRESENT"],
                    p_missing=probs["MISSING"],
                    p_background=probs["BACKGROUND"],
                    score=float(score),
                    white_ratio=evidence["white_ratio"],
                    sleeve_blob=evidence["sleeve_blob"],
                    edge_density=evidence["edge_density"],
                    rule=rule,
                )
            )

    detections: List[MicroDetection] = []
    for band in by_band:
        candidates = sorted(by_band[band], key=lambda d: d.score, reverse=True)
        kept: List[MicroDetection] = []
        for det in candidates:
            overlap_idx = next(
                (
                    idx
                    for idx, old in enumerate(kept)
                    if box_iou((det.x, det.y, det.w, det.h), (old.x, old.y, old.w, old.h)) > cfg.nms_iou
                ),
                None,
            )
            if overlap_idx is not None:
                old = kept[overlap_idx]
                can_replace = (
                    det.state == "MISSING"
                    and old.state == "PRESENT"
                    and (
                        (det.rule == "model" and det.p_missing >= 0.50)
                        or det.p_missing >= 0.70
                    )
                )
                can_restore_present = (
                    det.state == "PRESENT"
                    and old.state == "MISSING"
                    and old.p_missing < cfg.replacement_missing_threshold
                    and (
                        det.rule == "solid_sleeve"
                        or (old.p_missing >= 0.38 and det.p_present >= 0.72)
                    )
                    and det.p_missing < 0.20
                )
                if can_replace:
                    kept[overlap_idx] = det
                elif can_restore_present:
                    kept[overlap_idx] = det
                continue
            kept.append(det)
            if len(kept) >= cfg.max_candidates_per_band:
                break
        detections.extend(sorted(kept, key=lambda d: d.x))
    return detections


def apply_event_filter(rows: Sequence[MicroDetection], cfg: MicroConfig) -> Tuple[List[MicroDetection], List[dict]]:
    rows = [MicroDetection(**asdict(row)) for row in rows]
    missing = [row for row in rows if row.state == "MISSING"]
    missing.sort(key=lambda row: (round(row.x / 120), row.band, row.time_sec))
    event_id = 1
    events: List[dict] = []
    groups: Dict[Tuple[int, str], List[MicroDetection]] = {}
    for row in missing:
        groups.setdefault((round(row.x / 120), row.band), []).append(row)
    for _, group_rows in groups.items():
        current: List[MicroDetection] = []
        last_t = None
        for row in group_rows:
            if last_t is None or row.time_sec - last_t <= cfg.event_gap_sec:
                current.append(row)
            else:
                event_id = close_event(current, rows, events, event_id, cfg)
                current = [row]
            last_t = row.time_sec
        event_id = close_event(current, rows, events, event_id, cfg)
    return rows, events


def close_event(current: List[MicroDetection], all_rows: List[MicroDetection], events: List[dict], event_id: int, cfg: MicroConfig) -> int:
    if len(current) < cfg.min_event_observations:
        for row in current:
            row.final_state = "FILTERED_SPIKE"
        return event_id
    for row in current:
        for target in all_rows:
            if target.time_sec == row.time_sec and target.band == row.band and target.x == row.x and target.y == row.y:
                target.event_id = event_id
                target.final_state = "MISSING"
    peak = max(current, key=lambda row: row.p_missing)
    events.append(
        {
            "event_id": event_id,
            "band": peak.band,
            "start_sec": float(current[0].time_sec),
            "end_sec": float(current[-1].time_sec),
            "observations": len(current),
            "peak_time_sec": float(peak.time_sec),
            "peak_x": int(peak.x),
            "peak_p_missing": float(peak.p_missing),
        }
    )
    return event_id + 1


def save_rows(path: Path, rows: Sequence[MicroDetection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(MicroDetection.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def state_color(state: str) -> Color:
    return {
        "PRESENT": (45, 220, 95),
        "MISSING": (35, 70, 255),
        "FILTERED_SPIKE": (0, 165, 255),
        "UNKNOWN": (0, 225, 255),
        "BACKGROUND": (150, 150, 150),
    }.get(state, (235, 235, 235))


def annotate_frame(frame: np.ndarray, detections: Sequence[MicroDetection], cfg: MicroConfig, time_sec: float) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (cfg.right_half_x, frame.shape[0]), (0, 0, 0), -1)
    for y1, y2, label in [(cfg.upper_y1, cfg.upper_y2, "UPPER FIXED SIDE MICRO BAND"), (cfg.lower_y1, cfg.lower_y2, "LOWER FIXED SIDE MICRO BAND")]:
        cv2.rectangle(overlay, (cfg.right_half_x, y1), (frame.shape[1] - 1, y2), (40, 80, 90), -1)
        cv2.rectangle(canvas, (cfg.right_half_x, y1), (frame.shape[1] - 1, y2), (0, 220, 255), 2)
        cv2.putText(canvas, label, (cfg.right_half_x + 20, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0)
    cv2.line(canvas, (cfg.right_half_x, 0), (cfg.right_half_x, frame.shape[0]), (0, 230, 255), 3)
    cv2.putText(canvas, "FIXED MICRO ROI SIDE-CAP DETECTOR", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (245, 250, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"t={time_sec:07.2f}s  scan: two fixed side bands only", (28, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 225, 240), 2, cv2.LINE_AA)

    for det in detections:
        color = state_color(det.final_state)
        cv2.rectangle(canvas, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 3)
        label = f"{det.band.upper()} {det.final_state}  miss={det.p_missing:.2f}"
        tx = max(cfg.right_half_x + 4, min(det.x, frame.shape[1] - 260))
        cv2.rectangle(canvas, (tx, max(0, det.y - 25)), (tx + 255, max(22, det.y - 3)), (0, 0, 0), -1)
        cv2.putText(canvas, label, (tx + 5, max(16, det.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    panel_x, panel_y, panel_w, panel_h = 26, 96, 560, 340
    panel = canvas.copy()
    cv2.rectangle(panel, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (10, 13, 20), -1)
    canvas = cv2.addWeighted(panel, 0.78, canvas, 0.22, 0)
    misses = [d for d in detections if d.final_state == "MISSING"]
    presents = [d for d in detections if d.final_state == "PRESENT"]
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 230, 255) if not misses else (35, 70, 255), 2)
    cv2.putText(canvas, "MICRO SIDE ROI DIAGNOSTICS", (panel_x + 24, panel_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 250, 255), 2)
    lines = [
        f"fixed bands: upper {cfg.upper_y1}-{cfg.upper_y2}, lower {cfg.lower_y1}-{cfg.lower_y2}",
        f"window: {cfg.win_w}x{cfg.win_h}, step={cfg.scan_step}px",
        f"present boxes: {len(presents)}",
        f"missing boxes: {len(misses)}",
        "decision: opaque beige sleeve only; transparent ports fail",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(canvas, line, (panel_x + 26, panel_y + 82 + idx * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 232, 245), 1, cv2.LINE_AA)
    return canvas


def save_timeline(rows: Sequence[MicroDetection], events: Sequence[dict], out_path: Path) -> None:
    width, height = 1800, 520
    img = Image.new("RGB", (width, height), (12, 15, 22))
    draw = ImageDraw.Draw(img)
    draw.text((34, 24), "Fixed Side Micro ROI Detection Timeline", fill=(235, 245, 255), font=pil_font(30))
    if not rows:
        img.save(out_path)
        return
    t_min, t_max = min(r.time_sec for r in rows), max(r.time_sec for r in rows)
    x0, y0, x1, y1 = 70, 110, width - 50, height - 70
    draw.rectangle((x0, y0, x1, y1), outline=(70, 82, 105))
    by_time: Dict[float, float] = {}
    for row in rows:
        by_time[row.time_sec] = max(by_time.get(row.time_sec, 0.0), row.p_missing if row.final_state in {"MISSING", "FILTERED_SPIKE"} else 0.0)
    points = []
    for t in sorted(by_time):
        x = x0 + (t - t_min) / max(1e-6, t_max - t_min) * (x1 - x0)
        y = y1 - min(1.0, by_time[t]) * (y1 - y0)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(255, 210, 40), width=2)
    for event in events:
        sx = x0 + (event["start_sec"] - t_min) / max(1e-6, t_max - t_min) * (x1 - x0)
        ex = x0 + (event["end_sec"] - t_min) / max(1e-6, t_max - t_min) * (x1 - x0)
        draw.rectangle((sx, y0, ex, y1), outline=(255, 75, 55), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_review_video(video_path: Path, model: ExtraTreesClassifier, cfg: MicroConfig, rows_by_time: Dict[float, List[MicroDetection]], out_path: Path) -> None:
    cap, fps, frame_count, width, height = open_video(video_path)
    duration = frame_count / fps if fps else 0.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), cfg.review_fps, (width, height))
    for t in tqdm(list(iter_times(0.0, duration, cfg.sample_step_sec)), desc="micro_review"):
        ok, frame, frame_idx = read_frame_at(cap, fps, t)
        if not ok or frame is None:
            continue
        detections = rows_by_time.get(round(t, 4))
        if detections is None:
            detections = scan_frame(frame, model, cfg, t, frame_idx)
        writer.write(annotate_frame(frame, detections, cfg, t))
    writer.release()
    cap.release()


def run(args: argparse.Namespace) -> None:
    cfg = MicroConfig(sample_step_sec=args.sample_step)
    video_path = Path(args.video)
    annotation_dir = Path(args.annotations)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals = out_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)

    model, train_report = train_model(annotation_dir, cfg, out_dir)
    model_path = out_dir / "side_micro_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"cfg": cfg, "model": model, "train_report": train_report}, f)

    cap, fps, frame_count, _, _ = open_video(video_path)
    duration = frame_count / fps if fps else 0.0
    rows: List[MicroDetection] = []
    for t in tqdm(list(iter_times(0.0, duration, cfg.sample_step_sec)), desc="micro_scan"):
        ok, frame, frame_idx = read_frame_at(cap, fps, t)
        if not ok or frame is None:
            continue
        rows.extend(scan_frame(frame, model, cfg, t, frame_idx))
    cap.release()
    rows, events = apply_event_filter(rows, cfg)
    save_rows(out_dir / "observations.csv", rows)
    with (out_dir / "events.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(events[0].keys()) if events else ["event_id", "band", "start_sec", "end_sec", "observations", "peak_time_sec", "peak_x", "peak_p_missing"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)

    save_timeline(rows, events, visuals / "01_micro_score_timeline.png")
    annotation_mosaic = annotation_dir / "side_roi_crops_mosaic.jpg"
    if annotation_mosaic.exists():
        shutil.copy2(annotation_mosaic, visuals / "00_training_crops_mosaic.jpg")
    rows_by_time: Dict[float, List[MicroDetection]] = {}
    for row in rows:
        rows_by_time.setdefault(round(row.time_sec, 4), []).append(row)
    save_review_video(video_path, model, cfg, rows_by_time, visuals / "02_micro_review_2fps.mp4")

    counts = {state: int(sum(row.final_state == state for row in rows)) for state in ["PRESENT", "MISSING", "UNKNOWN", "FILTERED_SPIKE"]}
    summary = {
        "video": str(video_path),
        "annotations": str(annotation_dir / "annotations.json"),
        "model": str(model_path),
        "config": asdict(cfg),
        "train_report": train_report,
        "observations": len(rows),
        "counts": counts,
        "events": len(events),
        "outputs": {
            "observations": str(out_dir / "observations.csv"),
            "events": str(out_dir / "events.csv"),
            "review_video": str(visuals / "02_micro_review_2fps.mp4"),
            "timeline": str(visuals / "01_micro_score_timeline.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "events": len(events), "review_video": str((visuals / "02_micro_review_2fps.mp4").resolve())}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed side micro ROI detector trained from user annotations.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--video", default="20241101_161258.mp4")
    run_p.add_argument("--annotations", default="cap_annotation_tool")
    run_p.add_argument("--out", default="side_micro_outputs")
    run_p.add_argument("--sample-step", type=float, default=0.5)
    run_p.set_defaults(func=run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
