from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import output_dir, range_sec, roi_xyxy, video_path
from .video import crop_roi, open_video, read_rows_csv


def plot_scores(pred_csv: Path, out_path: Path) -> None:
    rows = read_rows_csv(pred_csv)
    if not rows:
        return
    t = np.array([float(r["time_sec"]) for r in rows], dtype=float)
    scores = np.array([float(r["score"]) for r in rows], dtype=float)
    threshold = float(rows[0]["threshold"])
    preds = np.array([r["pred"].upper() for r in rows])
    plt.figure(figsize=(11, 4))
    plt.plot(t, scores, label="score", color="#1f77b4", linewidth=1.4)
    plt.axhline(threshold, color="#d62728", linestyle="--", label="threshold")
    ng = preds == "NG"
    if np.any(ng):
        plt.scatter(t[ng], scores[ng], color="#d62728", s=16, label="NG")
    plt.xlabel("time (s)")
    plt.ylabel("anomaly score")
    plt.title("Anomaly score over time")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _overlay_heatmap(frame: np.ndarray, roi: tuple[int, int, int, int], heatmap: np.ndarray | None) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = roi
    if heatmap is not None:
        heat = cv2.resize(heatmap, (x2 - x1, y2 - y1))
        color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        out[y1:y2, x1:x2] = cv2.addWeighted(out[y1:y2, x1:x2], 0.65, color, 0.35, 0)
    return out


def draw_prediction(
    frame_bgr: np.ndarray,
    row: dict,
    roi: tuple[int, int, int, int],
    heatmap: np.ndarray | None = None,
) -> np.ndarray:
    out = _overlay_heatmap(frame_bgr, roi, heatmap)
    x1, y1, x2, y2 = roi
    pred = row["pred"].upper()
    color = (0, 220, 0) if pred == "OK" else (0, 0, 255)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    label = f'{pred}  score={float(row["score"]):.3f}  thr={float(row["threshold"]):.3f}'
    cv2.rectangle(out, (x1, max(0, y1 - 42)), (min(out.shape[1], x1 + 650), y1), color, -1)
    cv2.putText(out, label, (x1 + 10, max(28, y1 - 13)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def _resize_cover(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    y = max(0, (resized.shape[0] - height) // 2)
    x = max(0, (resized.shape[1] - width) // 2)
    return resized[y : y + height, x : x + width]


def _resize_contain(image: np.ndarray, size: tuple[int, int], fill=(13, 21, 33)) -> np.ndarray:
    width, height = size
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def _blend_rect(image: np.ndarray, xyxy: tuple[int, int, int, int], color: tuple[int, int, int], alpha: float) -> None:
    x1, y1, x2, y2 = xyxy
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    image[:] = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def _panel(image: np.ndarray, xyxy: tuple[int, int, int, int], title: str = "") -> None:
    x1, y1, x2, y2 = xyxy
    _blend_rect(image, xyxy, (42, 30, 15), 0.88)
    cv2.rectangle(image, (x1, y1), (x2, y2), (118, 86, 56), 1)
    if title:
        cv2.putText(image, title, (x1 + 16, y1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (210, 224, 242), 2)


def _put_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_metric_box(
    image: np.ndarray,
    xyxy: tuple[int, int, int, int],
    label: str,
    value: str,
    accent: tuple[int, int, int] = (56, 189, 248),
) -> None:
    x1, y1, x2, y2 = xyxy
    _blend_rect(image, xyxy, (36, 25, 14), 0.92)
    cv2.rectangle(image, (x1, y1), (x2, y2), accent, 1)
    _put_text(image, label.upper(), (x1 + 12, y1 + 25), 0.48, (148, 163, 184), 1)
    _put_text(image, value, (x1 + 12, y2 - 17), 0.7, (241, 245, 249), 2)


def _score_chart(
    image: np.ndarray,
    xyxy: tuple[int, int, int, int],
    times: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    current_time: float,
) -> None:
    x1, y1, x2, y2 = xyxy
    _panel(image, xyxy, "Realtime Score Trace")
    plot_x1, plot_y1, plot_x2, plot_y2 = x1 + 46, y1 + 48, x2 - 22, y2 - 32
    cv2.rectangle(image, (plot_x1, plot_y1), (plot_x2, plot_y2), (98, 73, 47), 1)
    if times.size == 0:
        return
    start_t, end_t = float(times.min()), float(times.max())
    if end_t <= start_t:
        end_t = start_t + 1.0
    y_max = max(float(scores.max()) if scores.size else 0.0, threshold * 1.25, 1e-6)
    y_min = 0.0
    for i in range(1, 4):
        y = int(plot_y2 - i * (plot_y2 - plot_y1) / 4)
        cv2.line(image, (plot_x1, y), (plot_x2, y), (52, 38, 24), 1)
    thr_y = int(plot_y2 - (threshold - y_min) / (y_max - y_min) * (plot_y2 - plot_y1))
    thr_y = max(plot_y1, min(plot_y2, thr_y))
    cv2.line(image, (plot_x1, thr_y), (plot_x2, thr_y), (37, 99, 235), 2)
    _put_text(image, "threshold", (plot_x1 + 8, max(plot_y1 + 16, thr_y - 7)), 0.43, (147, 197, 253), 1)

    points = []
    for t, s in zip(times, scores):
        px = int(plot_x1 + (float(t) - start_t) / (end_t - start_t) * (plot_x2 - plot_x1))
        py = int(plot_y2 - (float(s) - y_min) / (y_max - y_min) * (plot_y2 - plot_y1))
        points.append((px, max(plot_y1, min(plot_y2, py))))
    if len(points) >= 2:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, (34, 211, 238), 2)
    cur_x = int(plot_x1 + (current_time - start_t) / (end_t - start_t) * (plot_x2 - plot_x1))
    cur_x = max(plot_x1, min(plot_x2, cur_x))
    cv2.line(image, (cur_x, plot_y1), (cur_x, plot_y2), (250, 204, 21), 2)
    _put_text(image, f"{start_t:.0f}s", (plot_x1, plot_y2 + 22), 0.45, (148, 163, 184), 1)
    _put_text(image, f"{end_t:.0f}s", (plot_x2 - 56, plot_y2 + 22), 0.45, (148, 163, 184), 1)


def _score_bar(
    image: np.ndarray,
    xyxy: tuple[int, int, int, int],
    score: float,
    threshold: float,
    pred: str,
) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(image, (x1, y1), (x2, y2), (52, 38, 24), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (118, 86, 56), 1)
    max_value = max(threshold * 1.5, score, 1e-6)
    fill_x = int(x1 + min(1.0, score / max_value) * (x2 - x1))
    color = (16, 185, 129) if pred == "OK" else (14, 165, 233)
    if pred == "NG":
        color = (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (fill_x, y2), color, -1)
    thr_x = int(x1 + min(1.0, threshold / max_value) * (x2 - x1))
    cv2.line(image, (thr_x, y1 - 6), (thr_x, y2 + 6), (147, 197, 253), 2)


def _advanced_frame(
    frame_bgr: np.ndarray,
    row: dict,
    rows: list[dict],
    roi: tuple[int, int, int, int],
    heatmap: np.ndarray | None,
    meta_duration: float,
    render_start: float,
    render_end: float,
) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), (32, 20, 10), dtype=np.uint8)
    _blend_rect(canvas, (0, 0, 1920, 1080), (42, 27, 13), 0.65)

    pred = row["pred"].upper()
    score = float(row["score"])
    threshold = float(row["threshold"])
    time_sec = float(row["time_sec"])
    frame_idx = int(row["frame_idx"])
    accent = (34, 197, 94) if pred == "OK" else (0, 0, 255)

    _put_text(canvas, "Industrial Cap Assembly Inspection", (34, 44), 0.94, (226, 232, 240), 2)
    _put_text(canvas, "Realtime anomaly visualization", (35, 74), 0.5, (148, 163, 184), 1)
    _put_text(canvas, f"t={time_sec:06.2f}s  frame={frame_idx}", (1530, 47), 0.58, (203, 213, 225), 1)

    main = frame_bgr.copy()
    cv2.rectangle(main, (roi[0], roi[1]), (roi[2], roi[3]), accent, 4)
    cv2.putText(main, "ROI", (roi[0] + 12, max(40, roi[1] - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, accent, 2)
    main_panel = _resize_cover(main, (1180, 664))
    canvas[96:760, 24:1204] = main_panel
    cv2.rectangle(canvas, (24, 96), (1204, 760), (118, 86, 56), 1)
    _put_text(canvas, "SOURCE VIEW", (46, 132), 0.55, (203, 213, 225), 1)

    processed = draw_prediction(frame_bgr, row, roi, heatmap)
    proc_panel = _resize_cover(processed, (660, 352))
    canvas[96:448, 1236:1896] = proc_panel
    cv2.rectangle(canvas, (1236, 96), (1896, 448), (118, 86, 56), 1)
    _put_text(canvas, "HEATMAP OVERLAY", (1258, 132), 0.55, (203, 213, 225), 1)

    roi_img = crop_roi(frame_bgr, roi)
    if heatmap is not None:
        heat = cv2.resize(heatmap, (roi_img.shape[1], roi_img.shape[0]))
        color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        roi_img = cv2.addWeighted(roi_img, 0.58, color, 0.42, 0)
    roi_panel = _resize_contain(roi_img, (398, 256))
    canvas[488:744, 1236:1634] = roi_panel
    cv2.rectangle(canvas, (1236, 488), (1634, 744), (118, 86, 56), 1)
    _put_text(canvas, "ROI ZOOM", (1258, 522), 0.55, (203, 213, 225), 1)

    _panel(canvas, (1654, 488, 1896, 744), "Live Decision")
    badge_color = (20, 83, 45) if pred == "OK" else (127, 29, 29)
    _blend_rect(canvas, (1680, 536, 1870, 624), badge_color, 0.96)
    cv2.rectangle(canvas, (1680, 536), (1870, 624), accent, 2)
    _put_text(canvas, pred, (1726, 598), 1.85, (255, 255, 255), 4)
    _put_text(canvas, f"score {score:.4f}", (1680, 668), 0.62, (226, 232, 240), 2)
    _put_text(canvas, f"threshold {threshold:.4f}", (1680, 704), 0.54, (148, 163, 184), 1)
    _score_bar(canvas, (1680, 720, 1870, 736), score, threshold, pred)

    _panel(canvas, (24, 784, 566, 1050), "Telemetry")
    _draw_metric_box(canvas, (48, 836, 210, 912), "method", row.get("method", "method"))
    _draw_metric_box(canvas, (226, 836, 388, 912), "latency", f'{float(row.get("latency_ms", 0.0)):.1f} ms')
    _draw_metric_box(canvas, (404, 836, 542, 912), "motion", row.get("motion") or "n/a")
    _draw_metric_box(canvas, (48, 934, 210, 1010), "sharpness", f'{float(row.get("sharpness", 0.0)):.0f}')
    _draw_metric_box(canvas, (226, 934, 388, 1010), "yellow", row.get("yellow_ratio", "n/a")[:7])
    _draw_metric_box(canvas, (404, 934, 542, 1010), "fps", "8.0")

    times = np.array([float(r["time_sec"]) for r in rows], dtype=float)
    scores = np.array([float(r["score"]) for r in rows], dtype=float)
    window_mask = (times >= render_start) & (times <= render_end)
    _score_chart(canvas, (590, 784, 1420, 1050), times[window_mask], scores[window_mask], threshold, time_sec)

    _panel(canvas, (1444, 784, 1896, 1050), "Run Progress")
    progress = (time_sec - render_start) / max(1e-6, render_end - render_start)
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(canvas, (1474, 874), (1866, 908), (52, 38, 24), -1)
    cv2.rectangle(canvas, (1474, 874), (int(1474 + progress * 392), 908), (238, 211, 34), -1)
    cv2.rectangle(canvas, (1474, 874), (1866, 908), (118, 86, 56), 1)
    _put_text(canvas, f"last 2 minutes  {progress * 100:05.1f}%", (1474, 850), 0.62, (226, 232, 240), 2)
    _put_text(canvas, f"window {render_start:.1f}s - {render_end:.1f}s", (1474, 946), 0.54, (148, 163, 184), 1)
    _put_text(canvas, f"video duration {meta_duration:.1f}s", (1474, 982), 0.54, (148, 163, 184), 1)

    return canvas


def render_demo(
    cfg: dict,
    pred_csv: Path,
    out_video: Path | None = None,
    start_sec: float | None = None,
    duration_sec: float | None = None,
    fps: float | None = None,
    advanced: bool = False,
    last_minutes: float | None = None,
) -> dict:
    rows = read_rows_csv(pred_csv)
    if not rows:
        raise RuntimeError(f"No prediction rows found: {pred_csv}")
    render_cfg = cfg.get("render", {})
    cap, meta = open_video(video_path(cfg))
    if last_minutes is not None:
        duration_sec = float(last_minutes) * 60.0
        start_sec = max(0.0, meta.duration_sec - duration_sec)
    else:
        start_sec = float(render_cfg.get("demo_start_sec", 0.0) if start_sec is None else start_sec)
        duration_sec = float(render_cfg.get("demo_duration_sec", 15.0) if duration_sec is None else duration_sec)
    out_video = out_video or (output_dir(cfg) / ("demo_advanced_last2min_8fps.mp4" if advanced else "demo_15s.mp4"))
    fps_out = float(fps if fps is not None else render_cfg.get("fps", 15))
    roi = roi_xyxy(cfg)

    row_by_frame = {int(r["frame_idx"]): r for r in rows}
    start_idx = max(0, int(round(start_sec * meta.fps)))
    end_idx = min(meta.frame_count - 1, int(round((start_sec + duration_sec) * meta.fps)))
    render_end = min(meta.duration_sec, start_sec + duration_sec)
    total_out_frames = max(1, int(round((render_end - start_sec) * fps_out)))

    writer = None
    written = 0
    try:
        for out_idx in range(total_out_frames):
            target_time = start_sec + out_idx / fps_out
            frame_idx = int(round(target_time * meta.fps))
            frame_idx = max(start_idx, min(end_idx, frame_idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            nearest_idx = min(row_by_frame, key=lambda k: abs(k - frame_idx))
            row = row_by_frame[nearest_idx]
            heatmap = None
            heat_path = row.get("heatmap_path") or ""
            if heat_path and Path(heat_path).exists():
                heatmap = cv2.imread(heat_path, cv2.IMREAD_GRAYSCALE)
            if advanced:
                side = _advanced_frame(frame, row, rows, roi, heatmap, meta.duration_sec, start_sec, render_end)
            else:
                processed = draw_prediction(frame, row, roi, heatmap)
                left = frame.copy()
                cv2.putText(left, "Original", (32, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(processed, "Processed", (32, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                side = np.concatenate([left, processed], axis=1)
            if writer is None:
                out_video.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video), fourcc, fps_out, (side.shape[1], side.shape[0]))
            writer.write(side)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return {
        "demo_video": str(out_video),
        "frames_written": written,
        "fps": fps_out,
        "advanced": advanced,
        "start_sec": start_sec,
        "duration_sec": duration_sec,
    }


def make_figures_for_predictions(pred_csv: Path, out_dir: Path) -> dict[str, str]:
    score_path = out_dir / "score_timeline.png"
    plot_scores(pred_csv, score_path)
    return {"score_timeline": str(score_path)}
