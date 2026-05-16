from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from defect_detection_pipeline import (
    CircleCandidate,
    Observation,
    TexturePatchCore,
    heatmap_overlay,
    inner_face_metrics,
    load_model_bundle,
    load_observations_csv,
    open_video,
    pil_font,
    read_frame_at,
)


RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]

BG: RGBA = (9, 12, 19, 255)
PANEL: RGBA = (17, 22, 33, 255)
PANEL_2: RGBA = (24, 29, 41, 255)
GRID: RGBA = (67, 81, 104, 255)
TEXT: RGBA = (232, 242, 252, 255)
MUTED: RGBA = (164, 178, 196, 255)
CYAN: RGBA = (0, 212, 255, 255)
GREEN: RGBA = (68, 224, 95, 255)
YELLOW: RGBA = (255, 211, 48, 255)
ORANGE: RGBA = (255, 134, 42, 255)
RED: RGBA = (255, 64, 58, 255)
PURPLE: RGBA = (162, 118, 255, 255)
BLUE: RGBA = (80, 145, 255, 255)


def rgba(color: RGB | RGBA, alpha: int = 255) -> RGBA:
    if len(color) == 4:
        return color  # type: ignore[return-value]
    return (color[0], color[1], color[2], alpha)  # type: ignore[index]


def as_rgb_image(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def save_canvas(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    as_rgb_image(img).save(out_path)


def text_size(draw: ImageDraw.ImageDraw, text: str, size: int) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=pil_font(size))
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    size: int = 22,
    fill: RGBA = TEXT,
    anchor: Optional[str] = None,
) -> None:
    draw.text(xy, text, font=pil_font(size), fill=fill, anchor=anchor)


def draw_panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: Optional[str] = None) -> None:
    draw.rounded_rectangle(box, radius=10, fill=PANEL, outline=(82, 98, 124, 255), width=2)
    if title:
        draw_text(draw, (box[0] + 22, box[1] + 20), title, 25, TEXT)


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    title: str,
    value: str,
    subtitle: str,
    accent: RGBA,
) -> None:
    draw.rounded_rectangle(box, radius=10, fill=PANEL_2, outline=rgba(accent, 210), width=2)
    draw.rectangle((box[0], box[1], box[0] + 8, box[3]), fill=accent)
    draw_text(draw, (box[0] + 24, box[1] + 18), title.upper(), 17, MUTED)
    draw_text(draw, (box[0] + 24, box[1] + 49), value, 35, accent)
    draw_text(draw, (box[0] + 24, box[1] + 94), subtitle, 16, MUTED)


def parse_events(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key == "observations":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def obs_value(obs: Observation, name: str) -> float:
    return float(getattr(obs, name))


def quantile_stats(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def event_id_for_time(time_sec: float, events: Sequence[dict]) -> int:
    for idx, event in enumerate(events, start=1):
        if float(event["start_sec"]) <= time_sec <= float(event["end_sec"]):
            return idx
    return 0


def state_name(obs: Observation, threshold: float) -> str:
    if obs.is_defect:
        return "defect"
    if obs.hybrid_score > threshold:
        return "filtered_spike"
    return "normal"


def state_color(obs: Observation, threshold: float) -> RGBA:
    state = state_name(obs, threshold)
    if state == "defect":
        return RED
    if state == "filtered_spike":
        return ORANGE
    return GREEN


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def make_model_helper(model: TexturePatchCore, cfg) -> TexturePatchCore:
    helper = TexturePatchCore(cfg)
    helper.scaler = model.scaler
    helper.nn = model.nn
    helper.memory = model.memory
    helper.threshold = model.threshold
    helper.patch_mask = model.patch_mask
    return helper


def collect_crop_records(
    video_path: Path,
    cfg,
    model: TexturePatchCore,
    observations: Sequence[Observation],
) -> List[dict]:
    cap, _, _, _, _ = open_video(video_path)
    helper = make_model_helper(model, cfg)
    records: List[dict] = []
    for obs in observations:
        ok, frame, _ = read_frame_at(cap, obs.time_sec)
        if not ok or frame is None:
            continue
        cand = CircleCandidate(
            obs.cx,
            obs.cy,
            obs.r,
            obs.detection_score,
            obs.yellow_ratio,
            obs.saturation,
            obs.value,
        )
        crop = helper.crop_circle(frame, cand)
        _, heat, patch_scores = helper.score(crop)
        _, _, texture = inner_face_metrics(crop)
        records.append(
            {
                "obs": obs,
                "crop": crop,
                "heat": heat,
                "patch_scores": patch_scores,
                "texture": float(texture),
                "penalty": float(max(0.0, obs.hybrid_score - obs.patchcore_score)),
                "margin": float(obs.hybrid_score - (model.threshold or 0.0)),
            }
        )
    cap.release()
    return records


def draw_event_shading(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    events: Sequence[dict],
    x_min: float,
    x_max: float,
) -> None:
    x0, y0, x1, y1 = box
    span = max(1e-6, x_max - x_min)
    for event in events:
        ex0 = x0 + int((float(event["start_sec"]) - x_min) / span * (x1 - x0))
        ex1 = x0 + int((float(event["end_sec"]) - x_min) / span * (x1 - x0))
        ex0 = max(x0, min(x1, ex0))
        ex1 = max(x0, min(x1, ex1))
        if ex1 > ex0:
            draw.rectangle((ex0, y0, ex1, y1), fill=(86, 25, 25, 100))


def project_xy(
    x: float,
    y: float,
    box: Tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[int, int]:
    x0, y0, x1, y1 = box
    px = x0 + int((x - x_min) / max(1e-6, x_max - x_min) * (x1 - x0))
    py = y1 - int((y - y_min) / max(1e-6, y_max - y_min) * (y1 - y0))
    return px, py


def draw_chart_frame(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    title: str,
    y_min: float,
    y_max: float,
    label_color: RGBA = TEXT,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(11, 15, 24, 255), outline=GRID, width=2)
    draw_text(draw, (x0 + 10, y0 - 34), title, 21, label_color)
    for i in range(1, 5):
        y = y0 + int((y1 - y0) * i / 5)
        draw.line((x0, y, x1, y), fill=(45, 55, 72, 255), width=1)
    for value in [y_min, (y_min + y_max) / 2, y_max]:
        py = y1 - int((value - y_min) / max(1e-6, y_max - y_min) * (y1 - y0))
        draw_text(draw, (x0 - 8, py), f"{value:.2f}", 14, MUTED, anchor="ra")


def draw_time_series(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    color: RGBA,
    events: Sequence[dict],
    defect_mask: np.ndarray,
    threshold_line: Optional[Tuple[float, str, RGBA]] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    ref_lines: Sequence[Tuple[float, str, RGBA]] = (),
) -> None:
    if xs.size == 0:
        return
    x_min, x_max = float(xs.min()), float(xs.max())
    lo = float(np.nanmin(ys)) if y_min is None else y_min
    hi = float(np.nanmax(ys)) if y_max is None else y_max
    if threshold_line is not None:
        lo = min(lo, threshold_line[0])
        hi = max(hi, threshold_line[0])
    for ref, _, _ in ref_lines:
        lo = min(lo, ref)
        hi = max(hi, ref)
    pad = max(0.04, (hi - lo) * 0.12)
    lo -= pad
    hi += pad
    draw_chart_frame(draw, box, title, lo, hi)
    draw_event_shading(draw, box, events, x_min, x_max)
    points = [project_xy(float(x), float(y), box, x_min, x_max, lo, hi) for x, y in zip(xs, ys)]
    if len(points) >= 2:
        draw.line(points, fill=color, width=3, joint="curve")
    x0, y0, x1, y1 = box
    if threshold_line is not None:
        value, label, line_color = threshold_line
        py = project_xy(x_min, value, box, x_min, x_max, lo, hi)[1]
        draw.line((x0, py, x1, py), fill=line_color, width=2)
        draw_text(draw, (x0 + 10, py - 25), label, 16, line_color)
    for value, label, line_color in ref_lines:
        py = project_xy(x_min, value, box, x_min, x_max, lo, hi)[1]
        draw.line((x0, py, x1, py), fill=rgba(line_color, 150), width=1)
        draw_text(draw, (x1 - 10, py - 22), label, 14, line_color, anchor="ra")
    for point, is_defect in zip(points, defect_mask):
        if is_defect:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=RED)
    draw_text(draw, (x0, y1 + 10), f"{x_min:.1f}s", 15, MUTED)
    draw_text(draw, (x1, y1 + 10), f"{x_max:.1f}s", 15, MUTED, anchor="ra")


def draw_histogram(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    series: Sequence[Tuple[str, np.ndarray, RGBA]],
    x_min: float,
    x_max: float,
    threshold: Optional[float] = None,
    bins: int = 34,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(11, 15, 24, 255), outline=GRID, width=2)
    edges = np.linspace(x_min, x_max, bins + 1)
    counts = []
    max_count = 1
    for _, vals, _ in series:
        hist, _ = np.histogram(vals, bins=edges)
        counts.append(hist.astype(np.float32))
        if hist.size:
            max_count = max(max_count, int(hist.max()))
    bar_w = max(1, int((x1 - x0) / bins))
    for sidx, (label, _, color) in enumerate(series):
        hist = counts[sidx]
        for i, count in enumerate(hist):
            h = int((count / max_count) * (y1 - y0 - 28))
            bx0 = x0 + i * bar_w + sidx * max(1, bar_w // max(1, len(series)))
            bx1 = x0 + (i + 1) * bar_w - 1
            if len(series) > 1:
                bx1 = bx0 + max(1, bar_w // len(series) - 1)
            draw.rectangle((bx0, y1 - h, bx1, y1), fill=rgba(color, 155))
        lx = x0 + 18 + sidx * 250
        draw.rectangle((lx, y0 + 15, lx + 18, y0 + 33), fill=color)
        draw_text(draw, (lx + 26, y0 + 10), label, 17, TEXT)
    if threshold is not None:
        tx = x0 + int((threshold - x_min) / max(1e-6, x_max - x_min) * (x1 - x0))
        draw.line((tx, y0, tx, y1), fill=TEXT, width=3)
        draw_text(draw, (tx + 8, y0 + 42), f"T={threshold:.3f}", 17, TEXT)
    for value in np.linspace(x_min, x_max, 5):
        px = x0 + int((value - x_min) / max(1e-6, x_max - x_min) * (x1 - x0))
        draw.line((px, y1, px, y1 + 6), fill=GRID, width=1)
        draw_text(draw, (px, y1 + 10), f"{value:.1f}", 14, MUTED, anchor="ma")


def save_threshold_separation_report(
    observations: Sequence[Observation],
    threshold: float,
    threshold_scores: Sequence[float],
    metrics: dict,
    out_path: Path,
) -> None:
    img = Image.new("RGBA", (1900, 1180), BG)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (42, 34), "Threshold Separation Report", 42, TEXT)
    draw_text(
        draw,
        (44, 86),
        "Hybrid anomaly score distribution, operating margin, and the sample that exposed the previous false negative.",
        21,
        MUTED,
    )

    normal_scores = np.asarray([o.hybrid_score for o in observations if state_name(o, threshold) == "normal"], dtype=np.float64)
    filtered_scores = np.asarray([o.hybrid_score for o in observations if state_name(o, threshold) == "filtered_spike"], dtype=np.float64)
    defect_scores = np.asarray([o.hybrid_score for o in observations if o.is_defect], dtype=np.float64)
    calib_scores = np.asarray(threshold_scores, dtype=np.float64)
    all_scores = np.asarray([o.hybrid_score for o in observations], dtype=np.float64)
    x_min = float(max(0.0, min(all_scores.min(), calib_scores.min() if calib_scores.size else all_scores.min()) - 0.25))
    x_max = float(max(all_scores.max(), calib_scores.max() if calib_scores.size else all_scores.max(), threshold) + 0.35)

    draw_histogram(
        draw,
        (70, 190, 1240, 640),
        [
            ("calibration normal", calib_scores, BLUE),
            ("accepted normal", normal_scores, GREEN),
            ("filtered spike", filtered_scores, ORANGE),
            ("event defect", defect_scores, RED),
        ],
        x_min,
        x_max,
        threshold,
    )
    draw_text(draw, (70, 146), "Score Density", 27, TEXT)

    card_y = 172
    separation = metrics["separation"]
    cards = [
        ("threshold", f"{threshold:.3f}", "current operating line", TEXT),
        ("normal p95", f"{separation.get('accepted_normal_p95', 0.0):.3f}", "accepted samples upper band", GREEN),
        ("defect min", f"{separation.get('defect_min', 0.0):.3f}", "lowest event score", RED),
        ("t=382.50 margin", f"{separation.get('sample_382_5_margin', 0.0):+.3f}", f"{metrics['counts'].get('filtered_spikes', 0)} isolated spikes filtered", ORANGE),
    ]
    for idx, (title, value, subtitle, accent) in enumerate(cards):
        draw_card(draw, (1300, card_y + idx * 138, 1830, card_y + idx * 138 + 110), title, value, subtitle, accent)

    draw_panel(draw, (70, 735, 890, 1084), "Margin Strip")
    rng = np.random.default_rng(7)
    margin_min = min(float((all_scores - threshold).min()), -1.0)
    margin_max = max(float((all_scores - threshold).max()), 1.0)
    strip = (130, 830, 835, 1010)
    draw.rectangle(strip, fill=(11, 15, 24, 255), outline=GRID, width=2)
    zero_x = strip[0] + int((0.0 - margin_min) / max(1e-6, margin_max - margin_min) * (strip[2] - strip[0]))
    draw.line((zero_x, strip[1], zero_x, strip[3]), fill=TEXT, width=2)
    for obs in observations:
        m = obs.hybrid_score - threshold
        x = strip[0] + int((m - margin_min) / max(1e-6, margin_max - margin_min) * (strip[2] - strip[0]))
        state = state_name(obs, threshold)
        if state == "normal":
            base_y = strip[1] + 42
        elif state == "filtered_spike":
            base_y = strip[1] + 92
        else:
            base_y = strip[1] + 146
        y = int(base_y + rng.normal(0, 17))
        color = state_color(obs, threshold)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=rgba(color, 185))
    draw_text(draw, (strip[0], strip[3] + 18), f"{margin_min:+.1f}", 15, MUTED)
    draw_text(draw, (zero_x, strip[3] + 18), "0 margin", 15, TEXT, anchor="ma")
    draw_text(draw, (strip[2], strip[3] + 18), f"{margin_max:+.1f}", 15, MUTED, anchor="ra")
    draw_text(draw, (strip[0] - 12, strip[1] + 42), "normal", 16, GREEN, anchor="ra")
    draw_text(draw, (strip[0] - 12, strip[1] + 92), "filtered", 16, ORANGE, anchor="ra")
    draw_text(draw, (strip[0] - 12, strip[1] + 146), "defect", 16, RED, anchor="ra")

    draw_panel(draw, (960, 735, 1830, 1084), "Score Decomposition At t=382.50s")
    sample = metrics["sample_382_5"]
    sx0, sy0, sx1, sy1 = 1020, 870, 1760, 950
    patch = float(sample.get("patchcore_score", 0.0))
    penalty = float(sample.get("color_texture_penalty", 0.0))
    hybrid = float(sample.get("hybrid_score", 0.0))
    vmax = max(threshold * 1.25, hybrid * 1.12, 1.0)
    patch_w = int((patch / vmax) * (sx1 - sx0))
    penalty_w = int((penalty / vmax) * (sx1 - sx0))
    draw.rectangle((sx0, sy0, sx1, sy1), fill=(36, 42, 55, 255))
    draw.rectangle((sx0, sy0, sx0 + patch_w, sy1), fill=CYAN)
    draw.rectangle((sx0 + patch_w, sy0, min(sx1, sx0 + patch_w + penalty_w), sy1), fill=ORANGE)
    tx = sx0 + int((threshold / vmax) * (sx1 - sx0))
    draw.line((tx, sy0 - 18, tx, sy1 + 18), fill=TEXT, width=3)
    draw_text(draw, (sx0, sy0 - 48), f"patchcore={patch:.3f}", 19, CYAN)
    draw_text(draw, (sx0 + 250, sy0 - 48), f"color/texture boost={penalty:.3f}", 19, ORANGE)
    draw_text(draw, (tx + 6, sy1 + 22), f"T={threshold:.3f}", 17, TEXT)
    draw_text(draw, (sx0, sy1 + 58), f"hybrid={hybrid:.3f}, margin={hybrid - threshold:+.3f}", 23, RED if hybrid > threshold else GREEN)

    save_canvas(img, out_path)


def save_multimetric_dashboard(
    observations: Sequence[Observation],
    events: Sequence[dict],
    threshold: float,
    records: Sequence[dict],
    out_path: Path,
) -> None:
    img = Image.new("RGBA", (2200, 1540), BG)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (42, 34), "Multi-Metric Temporal Dashboard", 42, TEXT)
    draw_text(draw, (44, 86), "Score, PatchCore distance, color evidence, and localization stability across every sampled inspection.", 21, MUTED)

    xs = np.asarray([o.time_sec for o in observations], dtype=np.float64)
    defect_mask = np.asarray([o.is_defect for o in observations], dtype=bool)
    hybrid = np.asarray([o.hybrid_score for o in observations], dtype=np.float64)
    patch = np.asarray([o.patchcore_score for o in observations], dtype=np.float64)
    yellow = np.asarray([o.yellow_ratio for o in observations], dtype=np.float64)
    sat = np.asarray([o.saturation for o in observations], dtype=np.float64)
    penalty = np.asarray([r["penalty"] for r in records], dtype=np.float64)
    cx = np.asarray([o.cx for o in observations], dtype=np.float64)

    y0 = 180
    h = 180
    gap = 70
    box_x0, box_x1 = 115, 2110
    draw_time_series(draw, (box_x0, y0, box_x1, y0 + h), xs, hybrid, "Hybrid score", YELLOW, events, defect_mask, (threshold, f"threshold {threshold:.3f}", TEXT), y_min=0.0)
    y0 += h + gap
    draw_time_series(draw, (box_x0, y0, box_x1, y0 + h), xs, patch, "PatchCore nearest-neighbor distance", CYAN, events, defect_mask, y_min=0.0)
    y0 += h + gap
    draw_time_series(draw, (box_x0, y0, box_x1, y0 + h), xs, penalty, "Color/texture anomaly boost", ORANGE, events, defect_mask, y_min=0.0)
    y0 += h + gap
    draw_time_series(draw, (box_x0, y0, box_x1, y0 + h), xs, yellow, "Yellow face ratio", GREEN, events, defect_mask, y_min=0.0, y_max=1.0, ref_lines=[(0.50, "healthy tendency", GREEN)])
    y0 += h + gap
    draw_time_series(draw, (box_x0, y0, box_x1, y0 + h), xs, sat, "Inner-face saturation", BLUE, events, defect_mask, y_min=0.0, y_max=max(0.7, float(sat.max()) + 0.05), ref_lines=[(0.30, "low saturation zone", ORANGE)])

    ribbon = (box_x0, 1392, box_x1, 1430)
    draw.rectangle(ribbon, fill=(14, 18, 26, 255), outline=GRID, width=2)
    x_min, x_max = float(xs.min()), float(xs.max())
    for idx, obs in enumerate(observations):
        x = ribbon[0] + int((obs.time_sec - x_min) / max(1e-6, x_max - x_min) * (ribbon[2] - ribbon[0]))
        color = state_color(obs, threshold)
        draw.line((x, ribbon[1], x, ribbon[3]), fill=rgba(color, 190), width=3)
    draw_text(draw, (box_x0, 1450), "state ribbon: green = accepted sample, orange = filtered spike, red = sustained defect event", 18, MUTED)

    drift = (1550, 38, 2110, 132)
    draw.rounded_rectangle(drift, radius=10, fill=PANEL_2, outline=BLUE, width=2)
    cx_shift = float(np.median(np.abs(cx - np.median(cx))))
    draw_text(draw, (drift[0] + 18, drift[1] + 18), "Localization jitter", 17, MUTED)
    draw_text(draw, (drift[0] + 18, drift[1] + 50), f"median |cx-med(cx)| = {cx_shift:.1f}px", 30, BLUE)

    save_canvas(img, out_path)


def render_heat_grid(heat: np.ndarray, size: int = 170) -> Image.Image:
    h = heat.astype(np.float32)
    scale = max(float(np.percentile(h[h > 0], 95)) if np.any(h > 0) else 1.0, 1e-6)
    norm = np.clip(h / scale, 0, 1)
    grid = cv2.resize(norm, (size, size), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    for i in range(1, heat.shape[0]):
        p = int(i * size / heat.shape[0])
        cv2.line(color, (0, p), (size, p), (20, 20, 20), 1)
        cv2.line(color, (p, 0), (p, size), (20, 20, 20), 1)
    return bgr_to_pil(color)


def pick_key_records(records: Sequence[dict], limit: int = 16) -> List[dict]:
    if not records:
        return []
    by_time = {round(float(r["obs"].time_sec), 2): r for r in records}
    preferred_times = [360.0, 370.0, 372.5, 374.0, 382.5, 397.0, 430.0, 439.5]
    selected: List[dict] = []
    seen = set()
    for t in preferred_times:
        if round(t, 2) in by_time:
            selected.append(by_time[round(t, 2)])
            seen.add(round(t, 2))
    top_defects = sorted([r for r in records if r["obs"].is_defect], key=lambda r: r["obs"].hybrid_score, reverse=True)
    hard_normals = sorted([r for r in records if not r["obs"].is_defect], key=lambda r: r["obs"].hybrid_score, reverse=True)
    for pool in [top_defects, hard_normals, list(records)]:
        for r in pool:
            t = round(float(r["obs"].time_sec), 2)
            if t not in seen:
                selected.append(r)
                seen.add(t)
            if len(selected) >= limit:
                return selected
    return selected[:limit]


def save_heatmap_contact_sheet(records: Sequence[dict], threshold: float, out_path: Path) -> None:
    selected = pick_key_records(records, 16)
    cols = 4
    tile_w, tile_h = 455, 430
    header = 100
    rows = int(math.ceil(max(1, len(selected)) / cols))
    img = Image.new("RGBA", (cols * tile_w + 70, rows * tile_h + header + 40), BG)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (42, 34), "PatchCore Heatmap Contact Sheet", 40, TEXT)
    draw_text(draw, (44, 82), "Selected normal, transition, and event samples with crop, overlay, and raw patch-distance grid.", 20, MUTED)

    for idx, record in enumerate(selected):
        obs: Observation = record["obs"]
        col = idx % cols
        row = idx // cols
        x = 35 + col * tile_w
        y = header + 18 + row * tile_h
        state = state_name(obs, threshold).replace("_", " ").upper()
        accent = state_color(obs, threshold)
        draw.rounded_rectangle((x, y, x + tile_w - 28, y + tile_h - 22), radius=10, fill=PANEL, outline=accent, width=3)
        crop = cv2.resize(record["crop"], (170, 170), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(heatmap_overlay(record["crop"], record["heat"], threshold), (170, 170), interpolation=cv2.INTER_AREA)
        img.paste(bgr_to_pil(crop), (x + 18, y + 54))
        img.paste(bgr_to_pil(overlay), (x + 214, y + 54))
        img.paste(render_heat_grid(record["heat"], 170), (x + 18, y + 236))
        draw_text(draw, (x + 18, y + 17), f"t={obs.time_sec:.2f}s   {state}", 23, accent)
        draw_text(draw, (x + 214, y + 236), f"score {obs.hybrid_score:.3f}", 20, TEXT)
        draw_text(draw, (x + 214, y + 266), f"margin {record['margin']:+.3f}", 20, RED if record["margin"] > 0 else GREEN)
        draw_text(draw, (x + 214, y + 298), f"patch {obs.patchcore_score:.3f}", 17, CYAN)
        draw_text(draw, (x + 214, y + 324), f"yellow {obs.yellow_ratio:.3f}", 17, YELLOW)
        draw_text(draw, (x + 214, y + 350), f"sat {obs.saturation:.3f}", 17, BLUE)
        draw_text(draw, (x + 18, y + 414), "crop        overlay        patch grid", 14, MUTED)

    save_canvas(img, out_path)


def save_spatiotemporal_gate_map(
    observations: Sequence[Observation],
    events: Sequence[dict],
    cfg,
    threshold: float,
    out_path: Path,
) -> None:
    img = Image.new("RGBA", (1900, 1280), BG)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (42, 34), "Spatiotemporal Gate Map", 42, TEXT)
    draw_text(draw, (44, 86), "Where each sampled object crossed the virtual gate, and whether localization drift explains the score.", 21, MUTED)

    xs = np.asarray([o.time_sec for o in observations], dtype=np.float64)
    defect_mask = np.asarray([o.is_defect for o in observations], dtype=bool)
    cx = np.asarray([o.cx for o in observations], dtype=np.float64)
    cy = np.asarray([o.cy for o in observations], dtype=np.float64)
    r = np.asarray([o.r for o in observations], dtype=np.float64)
    det = np.asarray([o.detection_score for o in observations], dtype=np.float64)

    draw_time_series(draw, (120, 180, 1780, 370), xs, cx, "Center X at gate", CYAN, events, defect_mask, (float(cfg.gate_x), f"gate x={cfg.gate_x}", TEXT), y_min=float(cfg.roi_x1), y_max=float(cfg.roi_x2))
    draw_time_series(draw, (120, 470, 1780, 660), xs, cy, "Center Y stability", BLUE, events, defect_mask, y_min=float(cfg.roi_y1), y_max=float(cfg.roi_y2))
    draw_time_series(draw, (120, 760, 880, 960), xs, r, "Detected radius", PURPLE, events, defect_mask, y_min=float(cfg.min_radius), y_max=float(cfg.max_radius))
    draw_time_series(draw, (1020, 760, 1780, 960), xs, det, "Circle detector confidence", GREEN, events, defect_mask, y_min=0.0, y_max=1.0)

    panel = (120, 1035, 1780, 1210)
    draw.rectangle(panel, fill=(11, 15, 24, 255), outline=GRID, width=2)
    draw_text(draw, (panel[0] + 18, panel[1] - 36), "ROI Spatial Scatter", 22, TEXT)
    sx0, sy0, sx1, sy1 = panel
    gate_px = sx0 + int((cfg.gate_x - cfg.roi_x1) / max(1, cfg.roi_x2 - cfg.roi_x1) * (sx1 - sx0))
    draw.line((gate_px, sy0, gate_px, sy1), fill=TEXT, width=2)
    for obs in observations:
        px = sx0 + int((obs.cx - cfg.roi_x1) / max(1, cfg.roi_x2 - cfg.roi_x1) * (sx1 - sx0))
        py = sy0 + int((obs.cy - cfg.roi_y1) / max(1, cfg.roi_y2 - cfg.roi_y1) * (sy1 - sy0))
        color = state_color(obs, threshold)
        rr = 7 if obs.is_defect else 4
        draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=rgba(color, 170))
    draw_text(draw, (gate_px + 8, sy0 + 10), "virtual gate", 15, TEXT)

    save_canvas(img, out_path)


def feature_matrix(records: Sequence[dict], threshold: float) -> Tuple[np.ndarray, List[str]]:
    features = []
    for record in records:
        obs: Observation = record["obs"]
        features.append(
            [
                obs.hybrid_score,
                obs.patchcore_score,
                record["penalty"],
                1.0 - obs.yellow_ratio,
                1.0 - obs.saturation,
                obs.value,
                obs.detection_score,
                abs(obs.cx - 900) / 300.0,
                abs(obs.cy - 470) / 220.0,
                obs.r / 135.0,
                record["texture"],
                max(0.0, obs.hybrid_score - threshold),
            ]
        )
    names = [
        "hybrid",
        "patchcore",
        "penalty",
        "low yellow",
        "low sat",
        "value",
        "detect",
        "x shift",
        "y shift",
        "radius",
        "texture",
        "evidence",
    ]
    return np.asarray(features, dtype=np.float64), names


def draw_radar(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    radius: int,
    labels: Sequence[str],
    groups: Sequence[Tuple[str, np.ndarray, RGBA]],
) -> None:
    cx, cy = center
    n = len(labels)
    angles = np.linspace(-math.pi / 2, 1.5 * math.pi, n, endpoint=False)
    for ring in [0.25, 0.50, 0.75, 1.0]:
        pts = [
            (cx + int(math.cos(a) * radius * ring), cy + int(math.sin(a) * radius * ring))
            for a in angles
        ]
        draw.line(pts + [pts[0]], fill=(55, 66, 85, 255), width=1)
    for label, a in zip(labels, angles):
        end = (cx + int(math.cos(a) * radius), cy + int(math.sin(a) * radius))
        draw.line((cx, cy, end[0], end[1]), fill=(55, 66, 85, 255), width=1)
        tx = cx + int(math.cos(a) * (radius + 34))
        ty = cy + int(math.sin(a) * (radius + 34))
        draw_text(draw, (tx, ty), label, 14, MUTED, anchor="mm")
    for name, values, color in groups:
        pts = [
            (cx + int(math.cos(a) * radius * float(v)), cy + int(math.sin(a) * radius * float(v)))
            for a, v in zip(angles, values)
        ]
        draw.polygon(pts, fill=rgba(color, 54), outline=color)
        draw.line(pts + [pts[0]], fill=color, width=3)
    ly = cy + radius + 24
    lx = cx - radius
    for idx, (name, _, color) in enumerate(groups):
        draw.rectangle((lx, ly + idx * 24, lx + 16, ly + idx * 24 + 16), fill=color)
        draw_text(draw, (lx + 26, ly + idx * 24 - 3), name, 14, TEXT)


def save_feature_drift_atlas(
    records: Sequence[dict],
    events: Sequence[dict],
    threshold: float,
    out_path: Path,
) -> None:
    if len(records) < 3:
        return
    img = Image.new("RGBA", (1900, 1220), BG)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (42, 34), "Unsupervised Feature Drift Atlas", 42, TEXT)
    draw_text(draw, (44, 86), "The sampled stream as a latent trajectory: green normal cluster, red event cloud, and radial anomaly fingerprints.", 21, MUTED)

    X, names = feature_matrix(records, threshold)
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=7)
    coords = pca.fit_transform(scaled)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    plot = (90, 170, 1205, 1080)
    draw.rectangle(plot, fill=(11, 15, 24, 255), outline=GRID, width=2)
    draw_text(draw, (plot[0], plot[1] - 40), f"PCA latent trajectory  PC1={pca.explained_variance_ratio_[0]*100:.1f}%  PC2={pca.explained_variance_ratio_[1]*100:.1f}%", 22, TEXT)

    def proj(pt: np.ndarray) -> Tuple[int, int]:
        return project_xy(float(pt[0]), float(pt[1]), plot, float(x_min), float(x_max), float(y_min), float(y_max))

    pts = [proj(pt) for pt in coords]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(94, 108, 132, 125), width=2)
    times = np.asarray([r["obs"].time_sec for r in records], dtype=np.float64)
    t_min, t_max = float(times.min()), float(times.max())
    for pt, record in zip(pts, records):
        obs: Observation = record["obs"]
        state = state_name(obs, threshold)
        if state == "defect":
            color = RED
            rr = 8
        elif state == "filtered_spike":
            color = ORANGE
            rr = 7
        else:
            frac = (obs.time_sec - t_min) / max(1e-6, t_max - t_min)
            color = (int(55 + 80 * frac), int(220 - 60 * frac), int(120 + 100 * frac), 255)
            rr = 4
        draw.ellipse((pt[0] - rr, pt[1] - rr, pt[0] + rr, pt[1] + rr), fill=color)

    defects = [idx for idx, r in enumerate(records) if state_name(r["obs"], threshold) == "defect"]
    normals = [idx for idx, r in enumerate(records) if state_name(r["obs"], threshold) == "normal"]
    if normals:
        c = coords[normals].mean(axis=0)
        p = proj(c)
        draw.ellipse((p[0] - 14, p[1] - 14, p[0] + 14, p[1] + 14), outline=GREEN, width=4)
        draw_text(draw, (p[0] + 18, p[1] - 10), "normal centroid", 17, GREEN)
    if defects:
        c = coords[defects].mean(axis=0)
        p = proj(c)
        draw.ellipse((p[0] - 16, p[1] - 16, p[0] + 16, p[1] + 16), outline=RED, width=4)
        draw_text(draw, (p[0] + 20, p[1] - 12), "defect centroid", 17, RED)

    radar_box = (1300, 170, 1815, 790)
    draw_panel(draw, radar_box, "Anomaly Fingerprint Radar")
    selected_idx = [0, 1, 2, 3, 4, 7, 8, 11]
    selected_labels = [names[i] for i in selected_idx]
    subset = X[:, selected_idx]
    q05 = np.quantile(subset, 0.05, axis=0)
    q95 = np.quantile(subset, 0.95, axis=0)
    normed = np.clip((subset - q05) / np.maximum(q95 - q05, 1e-6), 0, 1)

    normal_mask = np.asarray([state_name(r["obs"], threshold) == "normal" for r in records], dtype=bool)
    defect_mask = np.asarray([state_name(r["obs"], threshold) == "defect" for r in records], dtype=bool)
    peak_idx = np.argsort([r["obs"].hybrid_score for r in records])[-max(3, min(20, len(records) // 10)) :]
    groups = []
    if normal_mask.any():
        groups.append(("normal baseline", normed[normal_mask].mean(axis=0), GREEN))
    if defect_mask.any():
        groups.append(("defect event", normed[defect_mask].mean(axis=0), RED))
    groups.append(("peak samples", normed[peak_idx].mean(axis=0), ORANGE))
    draw_radar(draw, (1560, 505), 205, selected_labels, groups)

    table = (1300, 835, 1815, 1080)
    draw_panel(draw, table, "Centroid Distance")
    if normals and defects:
        dist = float(np.linalg.norm(coords[defects].mean(axis=0) - coords[normals].mean(axis=0)))
        draw_text(draw, (table[0] + 26, table[1] + 80), f"{dist:.3f}", 48, ORANGE)
        draw_text(draw, (table[0] + 26, table[1] + 138), "distance between normal and defect centroids", 18, MUTED)
    if events:
        event = events[0]
        draw_text(draw, (table[0] + 26, table[1] + 190), f"event: {event['start_sec']:.1f}s -> {event['end_sec']:.1f}s", 22, RED)

    save_canvas(img, out_path)


def save_event_forensics_board(
    records: Sequence[dict],
    events: Sequence[dict],
    threshold: float,
    out_path: Path,
) -> None:
    if not records:
        return
    img = Image.new("RGBA", (2200, 1480), BG)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (42, 34), "Event Forensics Board", 42, TEXT)
    draw_text(draw, (44, 86), "One-page investigation view: onset, user-flagged frame, peak defect, evidence decomposition, and recovery boundary.", 21, MUTED)

    event = events[0] if events else None
    if event:
        draw_card(draw, (70, 150, 420, 260), "event start", f"{event['start_sec']:.1f}s", "first sustained defect sample", RED)
        draw_card(draw, (450, 150, 800, 260), "event end", f"{event['end_sec']:.1f}s", "last sustained defect sample", ORANGE)
        draw_card(draw, (830, 150, 1180, 260), "duration", f"{event['duration_sec']:.1f}s", f"{int(event['observations'])} sampled observations", YELLOW)
        draw_card(draw, (1210, 150, 1560, 260), "peak score", f"{event['peak_hybrid_score']:.3f}", f"at {event['peak_time_sec']:.1f}s", RED)
        draw_card(draw, (1590, 150, 1940, 260), "threshold", f"{threshold:.3f}", "operating line", TEXT)

    by_time = {round(float(r["obs"].time_sec), 2): r for r in records}
    def nearest_record(target: float) -> dict:
        return min(records, key=lambda r: abs(float(r["obs"].time_sec) - target))

    key_times = [370.0]
    if event:
        key_times.extend([float(event["start_sec"]), 382.5, float(event["peak_time_sec"]), float(event["end_sec"])])
    else:
        key_times.extend([382.5, records[len(records) // 2]["obs"].time_sec, records[-1]["obs"].time_sec])
    key_records = []
    seen = set()
    for t in key_times:
        rec = by_time.get(round(t, 2), nearest_record(t))
        rt = round(float(rec["obs"].time_sec), 2)
        if rt not in seen:
            key_records.append(rec)
            seen.add(rt)
    key_records = key_records[:5]

    y = 330
    tile_w = 410
    for idx, rec in enumerate(key_records):
        obs: Observation = rec["obs"]
        x = 70 + idx * tile_w
        state = state_name(obs, threshold).replace("_", " ").upper()
        accent = state_color(obs, threshold)
        draw.rounded_rectangle((x, y, x + tile_w - 22, y + 410), radius=10, fill=PANEL, outline=accent, width=3)
        crop = cv2.resize(rec["crop"], (168, 168), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(heatmap_overlay(rec["crop"], rec["heat"], threshold), (168, 168), interpolation=cv2.INTER_AREA)
        img.paste(bgr_to_pil(crop), (x + 20, y + 58))
        img.paste(bgr_to_pil(overlay), (x + 206, y + 58))
        draw_text(draw, (x + 20, y + 18), f"t={obs.time_sec:.2f}s", 25, accent)
        draw_text(draw, (x + 20, y + 248), state, 22, accent)
        lines = [
            f"score: {obs.hybrid_score:.3f} ({rec['margin']:+.3f})",
            f"patch: {obs.patchcore_score:.3f}",
            f"boost: {rec['penalty']:.3f}",
            f"yellow/sat: {obs.yellow_ratio:.3f} / {obs.saturation:.3f}",
        ]
        for li, line in enumerate(lines):
            draw_text(draw, (x + 20, y + 284 + li * 28), line, 17, TEXT if li == 0 else MUTED)

    # Evidence decomposition.
    panel = (70, 820, 2070, 1340)
    draw_panel(draw, panel, "Evidence Decomposition Over Time")
    plot = (150, 930, 1990, 1240)
    draw.rectangle(plot, fill=(11, 15, 24, 255), outline=GRID, width=2)
    xs = np.asarray([r["obs"].time_sec for r in records], dtype=np.float64)
    patch = np.asarray([r["obs"].patchcore_score for r in records], dtype=np.float64)
    penalty = np.asarray([r["penalty"] for r in records], dtype=np.float64)
    total = patch + penalty
    x_min, x_max = float(xs.min()), float(xs.max())
    y_max = max(float(total.max()), threshold * 1.2, 1.0)
    draw_event_shading(draw, plot, events, x_min, x_max)

    base_y = plot[3]
    patch_pts = [project_xy(float(x), float(y), plot, x_min, x_max, 0.0, y_max) for x, y in zip(xs, patch)]
    total_pts = [project_xy(float(x), float(y), plot, x_min, x_max, 0.0, y_max) for x, y in zip(xs, total)]
    patch_area = [(plot[0], base_y)] + patch_pts + [(plot[2], base_y)]
    total_area = patch_pts + list(reversed(total_pts))
    if len(patch_area) > 2:
        draw.polygon(patch_area, fill=rgba(CYAN, 75))
        draw.line(patch_pts, fill=CYAN, width=3)
    if len(total_area) > 2:
        draw.polygon(total_area, fill=rgba(ORANGE, 75))
        draw.line(total_pts, fill=YELLOW, width=3)
    th_y = project_xy(x_min, threshold, plot, x_min, x_max, 0.0, y_max)[1]
    draw.line((plot[0], th_y, plot[2], th_y), fill=TEXT, width=2)
    draw_text(draw, (plot[0] + 8, th_y - 28), f"threshold {threshold:.3f}", 16, TEXT)
    draw.rectangle((1550, 850, 1568, 868), fill=CYAN)
    draw_text(draw, (1578, 846), "PatchCore distance", 17, TEXT)
    draw.rectangle((1550, 884, 1568, 902), fill=ORANGE)
    draw_text(draw, (1578, 880), "Color/texture boost", 17, TEXT)

    save_canvas(img, out_path)


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for idx in range(values.size):
        lo = max(0, idx - window + 1)
        out[idx] = float(values[lo : idx + 1].sum())
    return out


def consecutive_count(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.size, dtype=np.float64)
    run = 0
    for idx, flag in enumerate(mask):
        run = run + 1 if bool(flag) else 0
        out[idx] = run
    return out


def save_cumulative_evidence_monitor(
    observations: Sequence[Observation],
    events: Sequence[dict],
    threshold: float,
    out_path: Path,
) -> None:
    img = Image.new("RGBA", (1900, 1180), BG)
    draw = ImageDraw.Draw(img)
    draw_text(draw, (42, 34), "Cumulative Evidence Monitor", 42, TEXT)
    draw_text(draw, (44, 86), "A production-friendly view of sustained evidence, rolling risk, and the two-sample event rule.", 21, MUTED)

    xs = np.asarray([o.time_sec for o in observations], dtype=np.float64)
    scores = np.asarray([o.hybrid_score for o in observations], dtype=np.float64)
    defect_mask = np.asarray([o.is_defect for o in observations], dtype=bool)
    raw_over = scores > threshold
    evidence = np.maximum(0.0, scores - threshold)
    roll5 = rolling_sum(evidence, 5)
    roll15 = rolling_sum(evidence, 15)
    consec = consecutive_count(raw_over)

    draw_time_series(draw, (120, 180, 1780, 380), xs, evidence, "Instant evidence = max(score - threshold, 0)", RED, events, defect_mask, y_min=0.0)
    draw_time_series(draw, (120, 500, 1780, 700), xs, roll5, "Rolling 5-sample evidence", ORANGE, events, defect_mask, y_min=0.0)
    draw_time_series(draw, (120, 820, 1780, 1020), xs, roll15, "Rolling 15-sample evidence", YELLOW, events, defect_mask, y_min=0.0)

    panel = (1220, 112, 1780, 148)
    draw.rectangle(panel, fill=PANEL_2, outline=GRID, width=2)
    if consec.size:
        draw_text(draw, (panel[0] + 16, panel[1] + 7), f"max consecutive above-threshold samples: {int(consec.max())}", 19, TEXT)
    draw_text(draw, (120, 1060), "Event rule used downstream: two or more consecutive over-threshold samples become a sustained defect event.", 19, MUTED)

    save_canvas(img, out_path)


def build_metrics(
    observations: Sequence[Observation],
    events: Sequence[dict],
    threshold: float,
    threshold_scores: Sequence[float],
    records: Sequence[dict],
) -> dict:
    normal = [o for o in observations if state_name(o, threshold) == "normal"]
    filtered = [o for o in observations if state_name(o, threshold) == "filtered_spike"]
    defect = [o for o in observations if state_name(o, threshold) == "defect"]
    sample = min(records, key=lambda r: abs(float(r["obs"].time_sec) - 382.5)) if records else None
    features = {
        "patchcore_score": [o.patchcore_score for o in observations],
        "yellow_ratio": [o.yellow_ratio for o in observations],
        "saturation": [o.saturation for o in observations],
        "detection_score": [o.detection_score for o in observations],
        "radius": [o.r for o in observations],
        "center_x": [o.cx for o in observations],
        "center_y": [o.cy for o in observations],
        "color_texture_penalty": [r["penalty"] for r in records],
        "texture": [r["texture"] for r in records],
    }
    hybrid = np.asarray([o.hybrid_score for o in observations], dtype=np.float64)
    correlations = {}
    for name, values in features.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == hybrid.size and arr.std() > 1e-9 and hybrid.std() > 1e-9:
            correlations[name] = float(np.corrcoef(arr, hybrid)[0, 1])

    normal_scores = [o.hybrid_score for o in normal]
    defect_scores = [o.hybrid_score for o in defect]
    accepted_normal_p95 = float(np.quantile(normal_scores, 0.95)) if normal_scores else 0.0
    defect_min = float(min(defect_scores)) if defect_scores else 0.0
    metrics = {
        "threshold": float(threshold),
        "counts": {
            "observations": len(observations),
            "accepted_normal": len(normal),
            "filtered_spikes": len(filtered),
            "event_defect": len(defect),
            "events": len(events),
            "raw_over_threshold": int(sum(o.hybrid_score > threshold for o in observations)),
            "near_threshold_abs_0_25": int(sum(abs(o.hybrid_score - threshold) <= 0.25 for o in observations)),
        },
        "score_stats": {
            "calibration_normal": quantile_stats(threshold_scores),
            "accepted_normal": quantile_stats(normal_scores),
            "filtered_spike": quantile_stats([o.hybrid_score for o in filtered]),
            "event_defect": quantile_stats(defect_scores),
            "all_observations": quantile_stats([o.hybrid_score for o in observations]),
        },
        "metric_stats": {name: quantile_stats(values) for name, values in features.items()},
        "correlation_with_hybrid_score": dict(sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)),
        "separation": {
            "accepted_normal_p95": accepted_normal_p95,
            "defect_min": defect_min,
            "defect_min_minus_accepted_normal_p95": float(defect_min - accepted_normal_p95),
            "sample_382_5_margin": float(sample["margin"]) if sample else 0.0,
        },
        "events": events,
        "sample_382_5": {
            "time_sec": float(sample["obs"].time_sec) if sample else None,
            "hybrid_score": float(sample["obs"].hybrid_score) if sample else None,
            "patchcore_score": float(sample["obs"].patchcore_score) if sample else None,
            "color_texture_penalty": float(sample["penalty"]) if sample else None,
            "yellow_ratio": float(sample["obs"].yellow_ratio) if sample else None,
            "saturation": float(sample["obs"].saturation) if sample else None,
            "texture": float(sample["texture"]) if sample else None,
            "margin": float(sample["margin"]) if sample else None,
            "is_defect": bool(sample["obs"].is_defect) if sample else None,
        },
    }
    return metrics


def write_analysis_tables(
    out_dir: Path,
    observations: Sequence[Observation],
    events: Sequence[dict],
    records: Sequence[dict],
    threshold: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_time = {round(float(r["obs"].time_sec), 4): r for r in records}
    with (out_dir / "score_margins.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "time_sec",
            "state",
            "event_id",
            "hybrid_score",
            "margin",
            "patchcore_score",
            "color_texture_penalty",
            "yellow_ratio",
            "saturation",
            "texture",
            "cx",
            "cy",
            "r",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for obs in observations:
            record = by_time.get(round(obs.time_sec, 4))
            writer.writerow(
                {
                    "time_sec": f"{obs.time_sec:.4f}",
                    "state": state_name(obs, threshold),
                    "event_id": event_id_for_time(obs.time_sec, events),
                    "hybrid_score": f"{obs.hybrid_score:.6f}",
                    "margin": f"{record['margin']:.6f}" if record else "",
                    "patchcore_score": f"{obs.patchcore_score:.6f}",
                    "color_texture_penalty": f"{record['penalty']:.6f}" if record else "",
                    "yellow_ratio": f"{obs.yellow_ratio:.6f}",
                    "saturation": f"{obs.saturation:.6f}",
                    "texture": f"{record['texture']:.6f}" if record else "",
                    "cx": obs.cx,
                    "cy": obs.cy,
                    "r": obs.r,
                }
            )
    with (out_dir / "event_forensics.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "event_id",
            "start_sec",
            "end_sec",
            "duration_sec",
            "observations",
            "peak_time_sec",
            "peak_hybrid_score",
            "peak_patchcore_score",
            "peak_saturation",
            "peak_yellow_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, event in enumerate(events, start=1):
            writer.writerow({"event_id": idx, **{k: event.get(k, "") for k in fields if k != "event_id"}})


def write_markdown_report(out_path: Path, metrics: dict, visual_dir: Path, analysis_dir: Path) -> None:
    lines = [
        "# Advanced Defect Analysis Report",
        "",
        f"- Threshold: `{metrics['threshold']:.3f}`",
        f"- Observations: `{metrics['counts']['observations']}`",
        f"- Accepted normal samples: `{metrics['counts']['accepted_normal']}`",
        f"- Filtered isolated spikes: `{metrics['counts']['filtered_spikes']}`",
        f"- Event samples: `{metrics['counts']['event_defect']}`",
        f"- Events: `{metrics['counts']['events']}`",
        f"- Near-threshold samples (+/-0.25): `{metrics['counts']['near_threshold_abs_0_25']}`",
        "",
        "## Separation",
        "",
        f"- Accepted normal p95: `{metrics['separation']['accepted_normal_p95']:.3f}`",
        f"- Defect minimum: `{metrics['separation']['defect_min']:.3f}`",
        f"- Defect minimum minus accepted-normal p95: `{metrics['separation']['defect_min_minus_accepted_normal_p95']:+.3f}`",
        f"- t=382.50s margin: `{metrics['separation']['sample_382_5_margin']:+.3f}`",
        "",
        "## Visuals",
        "",
    ]
    for path in sorted(visual_dir.glob("*.png")) + sorted(visual_dir.glob("*.jpg")):
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Data Tables",
            "",
            f"- `{(analysis_dir / 'advanced_metrics.json').name}`",
            f"- `{(analysis_dir / 'score_margins.csv').name}`",
            f"- `{(analysis_dir / 'event_forensics.csv').name}`",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    out_root = Path(args.out)
    visual_dir = out_root / "advanced_visuals"
    analysis_dir = out_root / "analysis"

    model, cfg, threshold_scores = load_model_bundle(Path(args.model_dir))
    observations = load_observations_csv(Path(args.observations))
    events = parse_events(Path(args.events))
    threshold = float(model.threshold or 0.0)

    records = collect_crop_records(video_path, cfg, model, observations)
    metrics = build_metrics(observations, events, threshold, threshold_scores, records)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "advanced_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_analysis_tables(analysis_dir, observations, events, records, threshold)

    save_threshold_separation_report(observations, threshold, threshold_scores, metrics, visual_dir / "08_threshold_separation_report.png")
    save_multimetric_dashboard(observations, events, threshold, records, visual_dir / "09_multimetric_temporal_dashboard.png")
    save_heatmap_contact_sheet(records, threshold, visual_dir / "10_patchcore_heatmap_contact_sheet.jpg")
    save_spatiotemporal_gate_map(observations, events, cfg, threshold, visual_dir / "11_spatiotemporal_gate_map.png")
    save_feature_drift_atlas(records, events, threshold, visual_dir / "12_unsupervised_feature_drift_atlas.png")
    save_event_forensics_board(records, events, threshold, visual_dir / "13_event_forensics_board.jpg")
    save_cumulative_evidence_monitor(observations, events, threshold, visual_dir / "14_cumulative_evidence_monitor.png")
    write_markdown_report(analysis_dir / "advanced_report.md", metrics, visual_dir, analysis_dir)

    print(f"advanced visuals: {visual_dir}")
    print(f"analysis tables: {analysis_dir}")
    print(f"threshold: {threshold:.3f}")
    print(f"observations: {len(observations)}  crop records: {len(records)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create advanced visual analytics for the conveyor defect detector.")
    parser.add_argument("--video", default="20241031_222226.mp4")
    parser.add_argument("--out", default="defect_outputs")
    parser.add_argument("--observations", default="defect_outputs/observations.csv")
    parser.add_argument("--events", default="defect_outputs/defect_events.csv")
    parser.add_argument("--model-dir", default="defect_outputs/model")
    return parser


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
