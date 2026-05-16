from __future__ import annotations

from pathlib import Path

import cv2

from .config import output_dir, range_sec, roi_xyxy, video_path
from .video import draw_roi, iter_samples, make_contact_sheet, open_video, save_json, write_rows_csv


def extract_frames(
    cfg: dict,
    split: str = "all",
    max_frames: int | None = None,
    save_images: bool = True,
) -> dict:
    out_dir = output_dir(cfg)
    vid = video_path(cfg)
    roi = roi_xyxy(cfg)
    frame_step = int(cfg.get("frame_step", 30))
    static_filter = cfg.get("static_filter", {})

    cap, meta = open_video(vid)
    cap.release()
    save_json(
        {
            "video_path": str(vid),
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "duration_sec": meta.duration_sec,
            "roi_xyxy": list(roi),
            "frame_step": frame_step,
            "static_filter": static_filter,
        },
        out_dir / "metadata" / "video_meta.json",
    )

    split_names = ["train", "val", "test"] if split == "all" else [split]
    all_rows: list[dict] = []
    contact_paths: list[Path] = []
    contact_labels: list[str] = []

    for split_name in split_names:
        start_sec, end_sec = range_sec(cfg, split_name)
        rows: list[dict] = []
        image_dir = out_dir / "frames" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        for sample in iter_samples(vid, start_sec, end_sec, frame_step, static_filter, max_frames=max_frames):
            frame_name = f"{split_name}_{sample.frame_idx:06d}.jpg"
            roi_name = f"{split_name}_{sample.frame_idx:06d}_roi.jpg"
            overlay_name = f"{split_name}_{sample.frame_idx:06d}_overlay.jpg"
            if save_images:
                cv2.imwrite(str(image_dir / frame_name), sample.frame_bgr)
                x1, y1, x2, y2 = roi
                cv2.imwrite(str(image_dir / roi_name), sample.frame_bgr[y1:y2, x1:x2])
                overlay = draw_roi(sample.frame_bgr, roi)
                overlay_path = image_dir / overlay_name
                cv2.imwrite(str(overlay_path), overlay)
                if len(contact_paths) < 18:
                    contact_paths.append(overlay_path)
                    contact_labels.append(f"{split_name} f={sample.frame_idx} t={sample.time_sec:.1f}s")
            row = {
                "video_id": cfg.get("video_id", "video"),
                "split": split_name,
                "frame_idx": sample.frame_idx,
                "time_sec": f"{sample.time_sec:.3f}",
                "motion": "" if sample.motion is None else f"{sample.motion:.4f}",
                "sharpness": f"{sample.sharpness:.4f}",
                "frame_path": str(image_dir / frame_name) if save_images else "",
                "roi_path": str(image_dir / roi_name) if save_images else "",
            }
            rows.append(row)
            all_rows.append(row)
        write_rows_csv(rows, out_dir / "metadata" / f"{split_name}_frames.csv")

    write_rows_csv(all_rows, out_dir / "metadata" / "frames.csv")
    make_contact_sheet(contact_paths, out_dir / "figures" / "roi_contact_sheet.jpg", contact_labels)
    return {
        "frames": len(all_rows),
        "metadata_csv": str(out_dir / "metadata" / "frames.csv"),
        "contact_sheet": str(out_dir / "figures" / "roi_contact_sheet.jpg"),
    }
