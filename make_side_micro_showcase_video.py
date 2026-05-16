from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import side_micro_roi_detector as micro


Color = Tuple[int, int, int]


@dataclass
class FrameStats:
    time_sec: float
    max_missing: float
    upper_missing: float
    lower_missing: float
    missing_count: int
    filtered_count: int
    present_count: int
    active_events: int


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def mix_color(a: Color, b: Color, t: float) -> Color:
    t = clamp01(t)
    return tuple(int(a[i] * (1.0 - t) + b[i] * t) for i in range(3))  # type: ignore[return-value]


def state_color(state: str) -> Color:
    return {
        "PRESENT": (60, 230, 90),
        "MISSING": (35, 65, 255),
        "FILTERED_SPIKE": (0, 165, 255),
        "UNKNOWN": (0, 220, 255),
    }.get(state, (220, 220, 220))


def draw_glow_rect(img: np.ndarray, rect: Tuple[int, int, int, int], color: Color, alpha: float = 0.22, border: int = 2) -> None:
    x1, y1, x2, y2 = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, border, cv2.LINE_AA)


def draw_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    scale: float,
    color: Color,
    thickness: int = 1,
    bg: Color | None = None,
) -> None:
    x, y = pos
    if bg is not None:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + 6), bg, -1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_bar(img: np.ndarray, label: str, value: float, rect: Tuple[int, int, int, int], color: Color) -> None:
    x1, y1, x2, y2 = rect
    value = clamp01(value)
    cv2.rectangle(img, (x1, y1), (x2, y2), (38, 38, 42), -1)
    cv2.rectangle(img, (x1, y1), (x1 + int((x2 - x1) * value), y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (90, 92, 100), 1, cv2.LINE_AA)
    draw_text(img, f"{label} {value:.2f}", (x1, y1 - 8), 0.46, (230, 236, 245), 1)


def draw_sparkline(img: np.ndarray, values: Sequence[float], rect: Tuple[int, int, int, int], color: Color, threshold: float = 0.30) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (22, 22, 28), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (88, 90, 96), 1, cv2.LINE_AA)
    ty = y2 - int(clamp01(threshold) * (y2 - y1))
    cv2.line(img, (x1, ty), (x2, ty), (70, 110, 125), 1, cv2.LINE_AA)
    if len(values) < 2:
        return
    clipped = [clamp01(v) for v in values[-160:]]
    points = []
    for idx, value in enumerate(clipped):
        x = x1 + int(idx / max(1, len(clipped) - 1) * (x2 - x1))
        y = y2 - int(value * (y2 - y1))
        points.append((x, y))
    for p1, p2 in zip(points, points[1:]):
        local_color = (35, 65, 255) if p2[1] < ty else color
        cv2.line(img, p1, p2, local_color, 2, cv2.LINE_AA)


def draw_risk_gauge(img: np.ndarray, center: Tuple[int, int], radius: int, risk: float, label: str) -> None:
    risk = clamp01(risk)
    cx, cy = center
    cv2.circle(img, center, radius, (35, 35, 44), 10, cv2.LINE_AA)
    end_angle = int(-210 + 300 * risk)
    color = mix_color((60, 230, 90), (35, 65, 255), risk)
    cv2.ellipse(img, center, (radius, radius), 0, -210, end_angle, color, 12, cv2.LINE_AA)
    cv2.circle(img, center, radius - 21, (17, 18, 24), -1, cv2.LINE_AA)
    draw_text(img, f"{risk * 100:04.1f}%", (cx - 64, cy + 12), 0.82, color, 2)
    draw_text(img, label, (cx - 76, cy + radius + 30), 0.48, (218, 226, 238), 1)


def crop_heatmap(crop_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    crop = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, sat, val = cv2.split(hsv)
    white = ((sat < 70) & (val > 115)).astype(np.uint8) * 255
    edges = cv2.Canny(gray, 70, 160)
    heat = np.zeros_like(crop)
    heat[:, :, 1] = white
    heat[:, :, 2] = edges
    heat[:, :, 0] = cv2.GaussianBlur(255 - white, (9, 9), 0)
    return cv2.addWeighted(crop, 0.42, heat, 0.58, 0)


def top_detections(detections: Sequence[micro.MicroDetection], count: int = 3) -> List[micro.MicroDetection]:
    return sorted(
        detections,
        key=lambda d: (
            1 if d.final_state == "MISSING" else 0,
            1 if d.final_state == "FILTERED_SPIKE" else 0,
            d.p_missing,
            d.edge_density,
        ),
        reverse=True,
    )[:count]


def stats_for_frame(time_sec: float, detections: Sequence[micro.MicroDetection]) -> FrameStats:
    upper = [d for d in detections if d.band == "upper"]
    lower = [d for d in detections if d.band == "lower"]
    missing = [d for d in detections if d.final_state == "MISSING"]
    filtered = [d for d in detections if d.final_state == "FILTERED_SPIKE"]
    present = [d for d in detections if d.final_state == "PRESENT"]
    return FrameStats(
        time_sec=time_sec,
        max_missing=max([d.p_missing for d in detections], default=0.0),
        upper_missing=max([d.p_missing for d in upper], default=0.0),
        lower_missing=max([d.p_missing for d in lower], default=0.0),
        missing_count=len(missing),
        filtered_count=len(filtered),
        present_count=len(present),
        active_events=len({d.event_id for d in missing if d.event_id}),
    )


def draw_right_detection_layer(frame: np.ndarray, detections: Sequence[micro.MicroDetection], cfg: micro.MicroConfig) -> None:
    overlay = frame.copy()
    for y1, y2, label in [
        (cfg.upper_y1, cfg.upper_y2, "UPPER FIXED SLEEVE BAND"),
        (cfg.lower_y1, cfg.lower_y2, "LOWER FIXED SLEEVE BAND"),
    ]:
        cv2.rectangle(overlay, (cfg.right_half_x, y1), (frame.shape[1] - 1, y2), (25, 55, 80), -1)
        cv2.rectangle(frame, (cfg.right_half_x, y1), (frame.shape[1] - 1, y2), (0, 220, 255), 2, cv2.LINE_AA)
        draw_text(frame, label, (cfg.right_half_x + 20, y1 - 10), 0.56, (0, 230, 255), 2)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
    cv2.line(frame, (cfg.right_half_x, 0), (cfg.right_half_x, frame.shape[0]), (0, 230, 255), 3, cv2.LINE_AA)

    for det in detections:
        color = state_color(det.final_state)
        cv2.rectangle(frame, (det.x, det.y), (det.x + det.w, det.y + det.h), color, 3, cv2.LINE_AA)
        label = f"{det.band.upper()} {det.final_state}  miss={det.p_missing:.2f}"
        tx = max(cfg.right_half_x + 8, min(det.x, frame.shape[1] - 315))
        ty = max(28, det.y - 10)
        draw_text(frame, label, (tx, ty), 0.48, color, 2, bg=(4, 8, 12))


def draw_lane_history(img: np.ndarray, stats: Sequence[FrameStats], idx: int, rect: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (22, 22, 28), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (85, 88, 98), 1, cv2.LINE_AA)
    draw_text(img, "BAND RISK STRIP - LAST 20S", (x1 + 12, y1 + 26), 0.46, (230, 236, 245), 1)
    rows = [("UPPER", "upper_missing", y1 + 48), ("LOWER", "lower_missing", y1 + 86)]
    start = max(0, idx - 160)
    window = stats[start : idx + 1]
    for label, attr, y in rows:
        draw_text(img, label, (x1 + 12, y + 18), 0.42, (200, 210, 225), 1)
        track_x1, track_x2 = x1 + 92, x2 - 14
        cv2.rectangle(img, (track_x1, y), (track_x2, y + 22), (36, 36, 42), -1)
        if window:
            bar_w = max(2, int((track_x2 - track_x1) / max(1, len(window))))
            for j, item in enumerate(window):
                value = clamp01(getattr(item, attr))
                color = mix_color((55, 210, 85), (30, 60, 255), value)
                bx1 = track_x1 + int(j / max(1, len(window)) * (track_x2 - track_x1))
                cv2.rectangle(img, (bx1, y), (min(track_x2, bx1 + bar_w), y + 22), color, -1)
        cv2.rectangle(img, (track_x1, y), (track_x2, y + 22), (80, 84, 92), 1, cv2.LINE_AA)


def draw_crop_cards(
    img: np.ndarray,
    frame: np.ndarray,
    detections: Sequence[micro.MicroDetection],
    rect: Tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = rect
    draw_glow_rect(img, rect, (70, 80, 105), 0.12, 1)
    draw_text(img, "TOP EVIDENCE CROPS", (x1 + 16, y1 + 30), 0.52, (236, 242, 250), 1)
    card_w, card_h = 276, 126
    for idx, det in enumerate(top_detections(detections, 3)):
        cx = x1 + 18 + idx * (card_w + 18)
        cy = y1 + 50
        crop = frame[det.y : det.y + det.h, det.x : det.x + det.w]
        if crop.size == 0:
            continue
        thumb = cv2.resize(crop, (92, 72), interpolation=cv2.INTER_AREA)
        heat = crop_heatmap(crop, (92, 72))
        color = state_color(det.final_state)
        cv2.rectangle(img, (cx, cy), (cx + card_w, cy + card_h), (22, 24, 30), -1)
        cv2.rectangle(img, (cx, cy), (cx + card_w, cy + card_h), color, 2, cv2.LINE_AA)
        img[cy + 12 : cy + 84, cx + 12 : cx + 104] = thumb
        img[cy + 12 : cy + 84, cx + 114 : cx + 206] = heat
        draw_text(img, f"{det.band.upper()} {det.final_state}", (cx + 12, cy + 105), 0.40, color, 1)
        draw_text(img, f"m={det.p_missing:.2f} e={det.edge_density:.2f}", (cx + 120, cy + 105), 0.38, (220, 226, 238), 1)


def draw_dashboard(
    frame: np.ndarray,
    detections: Sequence[micro.MicroDetection],
    stats: Sequence[FrameStats],
    idx: int,
    cfg: micro.MicroConfig,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    canvas = frame.copy()
    left_w = cfg.right_half_x
    left = canvas[:, :left_w]

    dark = np.zeros_like(left)
    dark[:, :] = (10, 12, 18)
    cv2.addWeighted(dark, 0.82, left, 0.18, 0, left)

    t = stats[idx].time_sec
    for x in range(0, left_w, 48):
        cv2.line(left, (x, 0), (x, left.shape[0]), (22, 28, 38), 1)
    for y in range(0, left.shape[0], 48):
        cv2.line(left, (0, y), (left_w, y), (22, 28, 38), 1)
    scan_y = int(((t - start_time) * 72) % left.shape[0])
    cv2.line(left, (0, scan_y), (left_w, scan_y), (0, 180, 255), 2, cv2.LINE_AA)

    current = stats[idx]
    risk = clamp01(current.max_missing + 0.10 * current.missing_count)
    verdict = "ALERT" if current.missing_count else ("WATCH" if current.filtered_count else "CLEAR")
    verdict_color = (35, 65, 255) if verdict == "ALERT" else ((0, 165, 255) if verdict == "WATCH" else (60, 230, 90))

    draw_text(left, "SIDE-SLEEVE DEFECT INTELLIGENCE", (34, 54), 0.78, (245, 248, 255), 2)
    draw_text(left, "LAST 60S SHOWCASE | 8 FPS | FIXED MICRO ROI", (36, 88), 0.48, (175, 205, 230), 1)
    draw_glow_rect(left, (34, 112, 310, 190), verdict_color, 0.18, 2)
    draw_text(left, verdict, (58, 166), 1.36, verdict_color, 3)
    draw_text(left, f"t={t:07.2f}s  demo={t - start_time:05.2f}/{end_time - start_time:04.1f}s", (340, 145), 0.58, (235, 238, 245), 1)
    draw_text(left, f"active events: {current.active_events}   missing boxes: {current.missing_count}", (340, 176), 0.50, (210, 220, 235), 1)

    draw_risk_gauge(left, (170, 320), 86, risk, "REAL-TIME DEFECT RISK")
    draw_bar(left, "MAX MISSING", current.max_missing, (330, 250, 840, 276), mix_color((60, 230, 90), (35, 65, 255), current.max_missing))
    draw_bar(left, "UPPER BAND", current.upper_missing, (330, 315, 840, 341), mix_color((60, 230, 90), (35, 65, 255), current.upper_missing))
    draw_bar(left, "LOWER BAND", current.lower_missing, (330, 380, 840, 406), mix_color((60, 230, 90), (35, 65, 255), current.lower_missing))

    spark_values = [item.max_missing for item in stats[: idx + 1]]
    draw_text(left, "LIVE MAX-MISSING TRACE", (42, 486), 0.52, (236, 242, 250), 1)
    draw_sparkline(left, spark_values, (40, 506, 890, 650), (255, 210, 40), threshold=0.30)
    draw_lane_history(left, stats, idx, (40, 680, 890, 820))
    draw_crop_cards(left, frame, detections, (40, 850, 890, 1010))

    rule_counts: Dict[str, int] = {}
    for det in detections:
        rule_counts[det.rule or "model"] = rule_counts.get(det.rule or "model", 0) + 1
    y = 1038
    draw_text(left, "RULE STACK:", (42, y), 0.44, (210, 220, 235), 1)
    text_x = 168
    for rule, count in sorted(rule_counts.items()):
        draw_text(left, f"{rule}={count}", (text_x, y), 0.42, (180, 220, 245), 1)
        text_x += 210

    draw_right_detection_layer(canvas, detections, cfg)
    return canvas


def write_rows(path: Path, rows: Sequence[micro.MicroDetection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else list(micro.MicroDetection.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    annotation_dir = Path(args.annotations)
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = micro.MicroConfig()
    cfg.review_fps = float(args.fps)
    cfg.sample_step_sec = 1.0 / float(args.fps)

    cap, src_fps, frame_count, width, height = micro.open_video(video_path)
    duration = frame_count / src_fps if src_fps else 0.0
    start_time = max(0.0, duration - float(args.seconds))
    end_time = duration
    times = list(micro.iter_times(start_time, end_time, cfg.sample_step_sec))
    cap.release()

    model, train_report = micro.train_model(annotation_dir, cfg, out_dir / "_showcase_training_cache")

    rows: List[micro.MicroDetection] = []
    cap, src_fps, _, _, _ = micro.open_video(video_path)
    for time_sec in tqdm(times, desc="showcase_scan"):
        ok, frame, frame_idx = micro.read_frame_at(cap, src_fps, time_sec)
        if not ok or frame is None:
            continue
        rows.extend(micro.scan_frame(frame, model, cfg, time_sec, frame_idx))
    cap.release()

    rows, events = micro.apply_event_filter(rows, cfg)
    rows_by_time: Dict[float, List[micro.MicroDetection]] = {}
    for row in rows:
        rows_by_time.setdefault(round(row.time_sec, 4), []).append(row)

    stats = [stats_for_frame(time_sec, rows_by_time.get(round(time_sec, 4), [])) for time_sec in times]
    write_rows(out_dir / "03_last60s_8fps_showcase_observations.csv", rows)
    with (out_dir / "03_last60s_8fps_showcase_events.json").open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height))
    cap, src_fps, _, _, _ = micro.open_video(video_path)
    snapshots = {}
    for idx, time_sec in enumerate(tqdm(times, desc="showcase_render")):
        ok, frame, frame_idx = micro.read_frame_at(cap, src_fps, time_sec)
        if not ok or frame is None:
            continue
        detections = rows_by_time.get(round(time_sec, 4), [])
        canvas = draw_dashboard(frame, detections, stats, idx, cfg, start_time, end_time)
        writer.write(canvas)
        rel = time_sec - start_time
        for target in [5.0, 30.0, 55.0]:
            if target not in snapshots and abs(rel - target) <= cfg.sample_step_sec / 2:
                snap_path = out_dir / f"03_showcase_snapshot_{int(target):02d}s.jpg"
                cv2.imwrite(str(snap_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
                snapshots[target] = str(snap_path)
    cap.release()
    writer.release()

    summary = {
        "video": str(video_path),
        "output": str(out_path),
        "start_time_sec": start_time,
        "end_time_sec": end_time,
        "fps": float(args.fps),
        "frames": len(times),
        "events": len(events),
        "counts": {
            "PRESENT": int(sum(row.final_state == "PRESENT" for row in rows)),
            "MISSING": int(sum(row.final_state == "MISSING" for row in rows)),
            "FILTERED_SPIKE": int(sum(row.final_state == "FILTERED_SPIKE" for row in rows)),
            "UNKNOWN": int(sum(row.final_state == "UNKNOWN" for row in rows)),
        },
        "train_report": train_report,
        "snapshots": snapshots,
    }
    (out_dir / "03_last60s_8fps_showcase_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an 8fps last-minute showcase video with advanced side-panel diagnostics.")
    parser.add_argument("--video", default="20241101_161258.mp4")
    parser.add_argument("--annotations", default="cap_annotation_tool")
    parser.add_argument("--out", default="side_micro_outputs/visuals/03_last60s_8fps_showcase.mp4")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=8.0)
    return parser


def main() -> None:
    parser = build_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
