from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or ROOT) / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = project_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(config_path.parent.parent if config_path.parent.name == "configs" else ROOT)
    return cfg


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    out = project_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def output_dir(cfg: dict[str, Any]) -> Path:
    return project_path(cfg.get("output_dir", "outputs/default"), ROOT)


def video_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["video_path"], ROOT)


def roi_xyxy(cfg: dict[str, Any]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = cfg["roi_xyxy"]
    return int(x1), int(y1), int(x2), int(y2)


def range_sec(cfg: dict[str, Any], split: str) -> tuple[float, float]:
    key = {
        "train": "train_range_sec",
        "val": "val_normal_range_sec",
        "test": "test_range_sec",
    }.get(split, split)
    start, end = cfg[key]
    return float(start), float(end)


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0
