from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .anomaly_dino import AnomalyDinoModel, DinoV2Extractor
from .baseline import YellowCapBaseline
from .config import ROOT, output_dir, range_sec, roi_xyxy, video_path
from .video import crop_roi, iter_samples, save_json, write_rows_csv


def _collect_train_rois(cfg: dict, max_frames: int | None = None) -> list[np.ndarray]:
    rois: list[np.ndarray] = []
    vid = video_path(cfg)
    roi = roi_xyxy(cfg)
    start, end = range_sec(cfg, "train")
    for sample in iter_samples(
        vid,
        start,
        end,
        int(cfg.get("frame_step", 30)),
        cfg.get("static_filter", {}),
        max_frames=max_frames,
    ):
        rois.append(crop_roi(sample.frame_bgr, roi))
    return rois


def build_memory_bank(cfg: dict, method: str, max_frames: int | None = None) -> dict[str, Any]:
    out_dir = output_dir(cfg)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_rois = _collect_train_rois(cfg, max_frames=max_frames)
    if method == "baseline":
        params = cfg.get("baseline", {})
        model = YellowCapBaseline(
            hsv_lower=params.get("hsv_lower", [12, 35, 60]),
            hsv_upper=params.get("hsv_upper", [48, 255, 255]),
            min_area_px=int(params.get("min_area_px", 25)),
        )
        stats = model.fit(train_rois, cfg.get("threshold_policy", {}))
        model.save(model_dir / "color_baseline.json", {"fit_stats": stats})
        return {"method": method, **stats, "model_path": str(model_dir / "color_baseline.json")}

    if method == "anomaly_dino":
        params = cfg.get("anomaly_dino", {})
        weight_path = Path(params.get("weight_path", "dinov2_vits14_pretrain.pth"))
        if not weight_path.is_absolute():
            weight_path = ROOT / weight_path
        extractor = DinoV2Extractor(
            model_name=params.get("model_name", "dinov2_vits14"),
            weight_path=weight_path,
            input_size=int(params.get("input_size", 448)),
            device=params.get("device", "auto"),
        )
        model = AnomalyDinoModel(
            extractor=extractor,
            top_percent=float(params.get("top_percent", 1.0)),
            nn_chunk_size=int(params.get("nn_chunk_size", 8192)),
            mask_patches=bool(params.get("mask_patches", False)),
        )
        stats = model.fit(
            train_rois,
            cfg.get("threshold_policy", {}),
            augment_rotations=bool(params.get("augment_rotations", False)),
            max_memory_patches=int(params.get("max_memory_patches", 80000)),
        )
        model.save(model_dir, {"fit_stats": stats})
        return {"method": method, **stats, "model_path": str(model_dir)}

    raise ValueError(f"Unknown method: {method}")


def _load_model(cfg: dict, method: str):
    out_dir = output_dir(cfg)
    model_dir = out_dir / "models"
    if method == "baseline":
        path = model_dir / "color_baseline.json"
        if not path.exists():
            build_memory_bank(cfg, method)
        return YellowCapBaseline.load(path)
    if method == "anomaly_dino":
        if not (model_dir / "anomaly_dino_meta.json").exists():
            build_memory_bank(cfg, method)
        return AnomalyDinoModel.load(model_dir, device=cfg.get("anomaly_dino", {}).get("device", "auto"))
    raise ValueError(f"Unknown method: {method}")


def infer_video(
    cfg: dict,
    method: str,
    split: str = "test",
    max_frames: int | None = None,
    write_heatmaps: bool = False,
) -> dict[str, Any]:
    out_dir = output_dir(cfg)
    pred_dir = out_dir / "predictions" / method
    heat_dir = pred_dir / "heatmaps"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if write_heatmaps:
        heat_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(cfg, method)
    vid = video_path(cfg)
    roi = roi_xyxy(cfg)
    start, end = range_sec(cfg, split)
    rows: list[dict] = []

    for sample in iter_samples(
        vid,
        start,
        end,
        int(cfg.get("frame_step", 30)),
        cfg.get("static_filter", {}),
        max_frames=max_frames,
    ):
        roi_img = crop_roi(sample.frame_bgr, roi)
        t0 = time.perf_counter()
        if method == "baseline":
            result = model.predict(roi_img)
            extra = {"yellow_ratio": f"{result.yellow_ratio:.8f}"}
        else:
            result = model.score_roi(roi_img)
            extra = {}
        latency_ms = (time.perf_counter() - t0) * 1000.0
        heatmap_path = ""
        if write_heatmaps:
            heatmap_path = str(heat_dir / f"{sample.frame_idx:06d}.png")
            cv2.imwrite(heatmap_path, result.heatmap)
        rows.append(
            {
                "video_id": cfg.get("video_id", "video"),
                "frame_idx": sample.frame_idx,
                "time_sec": f"{sample.time_sec:.3f}",
                "roi_xyxy": ",".join(str(v) for v in roi),
                "method": method,
                "score": f"{result.score:.8f}",
                "threshold": f"{model.threshold:.8f}",
                "pred": result.pred,
                "latency_ms": f"{latency_ms:.3f}",
                "motion": "" if sample.motion is None else f"{sample.motion:.4f}",
                "sharpness": f"{sample.sharpness:.4f}",
                "heatmap_path": heatmap_path,
                **extra,
            }
        )

    pred_path = pred_dir / f"{split}_predictions.csv"
    write_rows_csv(rows, pred_path)
    summary = {
        "method": method,
        "split": split,
        "count": len(rows),
        "prediction_csv": str(pred_path),
        "avg_latency_ms": float(np.mean([float(r["latency_ms"]) for r in rows])) if rows else 0.0,
        "fps_estimate": 1000.0 / float(np.mean([float(r["latency_ms"]) for r in rows])) if rows else 0.0,
    }
    save_json(summary, pred_dir / f"{split}_summary.json")
    return summary
