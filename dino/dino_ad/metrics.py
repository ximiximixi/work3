from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .video import read_rows_csv, save_json, write_rows_csv


def write_label_template(pred_csv: Path, out_csv: Path) -> None:
    rows = read_rows_csv(pred_csv)
    template = [
        {
            "video_id": r.get("video_id", ""),
            "frame_idx": r.get("frame_idx", ""),
            "time_sec": r.get("time_sec", ""),
            "label": "",
            "note": "",
        }
        for r in rows
    ]
    write_rows_csv(template, out_csv)


def _load_labels(labels_csv: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip().upper()
            if label in {"OK", "NG"}:
                labels[int(row["frame_idx"])] = label
    return labels


def evaluate_predictions(pred_csv: Path, labels_csv: Path | None, out_json: Path) -> dict[str, Any]:
    pred_rows = read_rows_csv(pred_csv)
    if labels_csv is None or not labels_csv.exists():
        template_path = out_json.with_name("labels_template.csv")
        write_label_template(pred_csv, template_path)
        result = {
            "status": "missing_labels",
            "message": "No labels were provided. Fill labels_template.csv with OK/NG and rerun evaluate.",
            "prediction_count": len(pred_rows),
            "labels_template": str(template_path),
        }
        save_json(result, out_json)
        return result

    labels = _load_labels(labels_csv)
    y_true: list[int] = []
    y_pred: list[int] = []
    used_rows = 0
    for row in pred_rows:
        frame_idx = int(row["frame_idx"])
        if frame_idx not in labels:
            continue
        y_true.append(1 if labels[frame_idx] == "NG" else 0)
        y_pred.append(1 if row["pred"].upper() == "NG" else 0)
        used_rows += 1
    if not y_true:
        result = {
            "status": "no_overlapping_labels",
            "prediction_count": len(pred_rows),
            "labeled_count": len(labels),
        }
        save_json(result, out_json)
        return result

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    latencies = [float(r["latency_ms"]) for r in pred_rows if r.get("latency_ms")]
    result = {
        "status": "ok",
        "used_rows": used_rows,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "fps": float(1000.0 / np.mean(latencies)) if latencies else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    save_json(result, out_json)
    return result
