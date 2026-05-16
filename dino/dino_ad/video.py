from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import VideoMeta


@dataclass
class FrameSample:
    frame_idx: int
    time_sec: float
    frame_bgr: np.ndarray
    motion: float | None
    sharpness: float
    split: str = ""


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    meta = VideoMeta(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, meta


def crop_roi(frame_bgr: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = roi_xyxy
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    return frame_bgr[y1:y2, x1:x2]


def draw_roi(frame_bgr: np.ndarray, roi_xyxy: tuple[int, int, int, int], color=(0, 255, 255)) -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = roi_xyxy
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
    return out


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _motion(gray: np.ndarray, prev_gray: np.ndarray | None) -> float | None:
    if prev_gray is None:
        return None
    return float(np.mean(cv2.absdiff(gray, prev_gray)))


def iter_samples(
    video_file: Path,
    start_sec: float,
    end_sec: float,
    frame_step: int,
    static_filter: dict | None = None,
    max_frames: int | None = None,
) -> Iterator[FrameSample]:
    cap, meta = open_video(video_file)
    start_idx = max(0, int(round(start_sec * meta.fps)))
    end_idx = min(meta.frame_count - 1, int(round(end_sec * meta.fps)))
    resize_width = int((static_filter or {}).get("resized_width", 320))
    use_static = bool((static_filter or {}).get("enabled", False))
    motion_threshold = float((static_filter or {}).get("motion_threshold", 999999.0))
    min_sharpness = float((static_filter or {}).get("min_sharpness", 0.0))

    prev_gray: np.ndarray | None = None
    yielded = 0
    try:
        for frame_idx in range(start_idx, end_idx + 1, max(1, int(frame_step))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            scale = resize_width / frame.shape[1]
            small = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sharp = _sharpness(gray)
            mot = _motion(gray, prev_gray)
            prev_gray = gray

            keep = True
            if use_static:
                if mot is not None and mot > motion_threshold:
                    keep = False
                if sharp < min_sharpness:
                    keep = False
            if not keep:
                continue

            yield FrameSample(
                frame_idx=frame_idx,
                time_sec=frame_idx / meta.fps if meta.fps else 0.0,
                frame_bgr=frame,
                motion=mot,
                sharpness=sharp,
            )
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        cap.release()


def write_rows_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_contact_sheet(
    image_paths: list[Path],
    out_path: Path,
    labels: list[str] | None = None,
    thumb_size: tuple[int, int] = (320, 180),
    columns: int = 3,
) -> None:
    if not image_paths:
        return
    labels = labels or [p.name for p in image_paths]
    rows = int(np.ceil(len(image_paths) / columns))
    cell_h = thumb_size[1] + 28
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail(thumb_size)
        x = (i % columns) * thumb_size[0]
        y = (i // columns) * cell_h
        sheet.paste(img, (x, y + 24))
        draw.text((x + 6, y + 6), labels[i][:56], fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
