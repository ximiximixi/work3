from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2


VIDEO_PATH = Path("20241101_161258.mp4")
OBS_PATH = Path("cap_defect_outputs/observations.csv")
TOOL_DIR = Path("cap_annotation_tool")
FRAME_DIR = TOOL_DIR / "frames"
RIGHT_HALF_X = 960

# About 20 frames: a few normal references, earlier suspected defects, and a
# dense block near the end where rare abnormal samples appear.
SELECTED_TIMES = [
    17.28,
    17.32,
    17.39,
    17.50,
    120.00,
    360.00,
    900.00,
    184.00,
    234.50,
    280.00,
    577.50,
    759.00,
    1009.50,
    1072.00,
    1074.00,
    1074.50,
    1075.00,
    1076.50,
    1082.50,
    1083.00,
    1083.50,
    1084.00,
]


def load_observations() -> List[Dict[str, str]]:
    if not OBS_PATH.exists():
        return []
    with OBS_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def nearest_predictions(rows: List[Dict[str, str]], time_sec: float, tolerance: float = 0.26) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            row_time = float(row["time_sec"])
        except (KeyError, ValueError):
            continue
        if abs(row_time - time_sec) > tolerance:
            continue
        try:
            top_x = int(float(row["top_anchor_x"]))
            top_y = int(float(row["top_anchor_y"]))
            top_w = int(float(row["top_anchor_w"]))
            top_h = int(float(row["top_anchor_h"]))
            bottom_x = int(float(row["bottom_anchor_x"]))
            bottom_y = int(float(row["bottom_anchor_y"]))
            bottom_w = int(float(row["bottom_anchor_w"]))
            bottom_h = int(float(row["bottom_anchor_h"]))
            sample_x1 = min(top_x, bottom_x)
            sample_y1 = min(top_y, bottom_y)
            sample_x2 = max(top_x + top_w, bottom_x + bottom_w)
            sample_y2 = max(top_y + top_h, bottom_y + bottom_h)
        except (KeyError, ValueError):
            continue
        out.append(
            {
                "sample_index": int(float(row.get("sample_index", 0) or 0)),
                "state": row.get("final_state", ""),
                "missing_side": row.get("missing_side", ""),
                "unknown_reason": row.get("unknown_reason", ""),
                "sample_box": {
                    "x": sample_x1,
                    "y": sample_y1,
                    "w": sample_x2 - sample_x1,
                    "h": sample_y2 - sample_y1,
                },
                "top_anchor": {"x": top_x, "y": top_y, "w": top_w, "h": top_h},
                "bottom_anchor": {"x": bottom_x, "y": bottom_y, "w": bottom_w, "h": bottom_h},
            }
        )
    out.sort(key=lambda item: item["sample_box"]["x"])
    return out


def time_bucket(time_sec: float) -> str:
    if time_sec < 180:
        return "normal_or_early_review"
    if time_sec >= 1070:
        return "late_rare_abnormal_review"
    return "mid_video_suspect_review"


def main() -> None:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    rows = load_observations()

    frames: List[Dict[str, Any]] = []
    for idx, time_sec in enumerate(SELECTED_TIMES, start=1):
        frame_idx = max(0, min(frame_count - 1, int(round(time_sec * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"skip failed frame at {time_sec:.2f}s")
            continue
        crop = frame[:, RIGHT_HALF_X:width]
        time_tag = f"{time_sec:07.2f}".replace(".", "p")
        filename = f"frame_{idx:02d}_t{time_tag}.jpg"
        out_path = FRAME_DIR / filename
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        predictions = nearest_predictions(rows, time_sec)
        frames.append(
            {
                "id": f"f{idx:02d}",
                "time_sec": time_sec,
                "frame_idx": frame_idx,
                "filename": f"frames/{filename}",
                "width": crop.shape[1],
                "height": crop.shape[0],
                "original_width": width,
                "original_height": height,
                "offset_x": RIGHT_HALF_X,
                "bucket": time_bucket(time_sec),
                "auto_predictions": predictions,
            }
        )
    cap.release()

    tasks = {
        "version": 1,
        "source_video": str(VIDEO_PATH),
        "fps": fps,
        "right_half_x": RIGHT_HALF_X,
        "instructions": [
            "Annotate only the right-half crop. Coordinates are stored in original video coordinates.",
            "Use NORMAL when visible caps/sleeves are complete.",
            "Use DEFECT when a visible required beige sleeve/cap is missing.",
            "Use UNKNOWN when the view is cropped, occluded, mirrored, or not enough evidence.",
        ],
        "label_schema": {
            "frame_state": ["UNLABELED", "NORMAL", "DEFECT", "UNKNOWN"],
            "object_type": ["sample", "fixed_region", "top", "bottom", "side_upper", "side_lower", "ignore"],
            "missing_parts": ["top", "bottom", "side_upper", "side_lower"],
        },
        "frames": frames,
    }
    (TOOL_DIR / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    annotations_path = TOOL_DIR / "annotations.json"
    if not annotations_path.exists():
        annotations = {
            "version": 1,
            "source_tasks": "tasks.json",
            "frames": {
                frame["id"]: {
                    "frame_state": "UNLABELED",
                    "missing_parts": [],
                    "notes": "",
                    "objects": [],
                }
                for frame in frames
            },
        }
        annotations_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"frames": len(frames), "tasks": str((TOOL_DIR / "tasks.json").resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
