from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video import load_json, save_json


class MissingTorchError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        raise MissingTorchError(
            "AnomalyDINO requires PyTorch. Install torch/torchvision, then rerun "
            "the command. The color baseline and visualization pipeline do not "
            "require PyTorch."
        ) from exc
    return torch


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def resize_pad_square(roi_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = roi_bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(roi_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - nh) // 2
    x = (size - nw) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


@dataclass
class DinoScore:
    score: float
    pred: str
    distances: np.ndarray
    heatmap: np.ndarray
    latency_ms: float = 0.0


class DinoV2Extractor:
    def __init__(self, model_name: str, weight_path: Path, input_size: int = 448, device: str = "auto"):
        torch = _require_torch()
        self.torch = torch
        self.model_name = model_name
        self.weight_path = Path(weight_path)
        self.input_size = int(input_size)
        if self.input_size % 14 != 0:
            raise ValueError("DINOv2 input_size must be a multiple of 14.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self._load_model().to(self.device).eval()

    def _load_model(self):
        torch = self.torch
        if not self.weight_path.exists():
            raise FileNotFoundError(f"DINOv2 weight file not found: {self.weight_path}")
        try:
            model = torch.hub.load("facebookresearch/dinov2", self.model_name, pretrained=False)
        except Exception as exc:
            raise RuntimeError(
                "Could not load DINOv2 architecture through torch.hub. "
                "Check network access or pre-cache facebookresearch/dinov2. "
                f"Requested model: {self.model_name}"
            ) from exc
        state = torch.load(str(self.weight_path), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[AnomalyDINO] Unexpected keys ignored: {len(unexpected)}")
        if missing:
            print(f"[AnomalyDINO] Missing keys: {len(missing)}")
        return model

    def _preprocess(self, roi_bgr: np.ndarray):
        torch = self.torch
        image = resize_pad_square(roi_bgr, self.input_size)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]
        return torch.from_numpy(chw).to(self.device)

    def extract(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        torch = self.torch
        with torch.no_grad():
            x = self._preprocess(roi_bgr)
            out = self.model.forward_features(x)
            if isinstance(out, dict) and "x_norm_patchtokens" in out:
                tokens = out["x_norm_patchtokens"]
            elif isinstance(out, dict) and "x_prenorm" in out:
                tokens = out["x_prenorm"][:, 1:]
            else:
                raise RuntimeError("DINOv2 forward_features did not expose patch tokens.")
            feats = tokens.squeeze(0).detach().float().cpu().numpy()
        feats = _normalize_rows(feats.astype(np.float32))
        grid = self.input_size // 14
        return feats, (grid, grid)


def pca_patch_mask(features: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    comp = centered @ vh[0]
    comp_img = comp.reshape(grid_shape)
    comp_norm = cv2.normalize(comp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(comp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = grid_shape
    center_label = binary[h // 2, w // 2]
    mask = binary == center_label
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return (mask_u8 > 0).reshape(-1)


class AnomalyDinoModel:
    def __init__(
        self,
        extractor: DinoV2Extractor,
        threshold: float = 0.0,
        top_percent: float = 1.0,
        nn_chunk_size: int = 8192,
        mask_patches: bool = False,
    ):
        self.extractor = extractor
        self.threshold = float(threshold)
        self.top_percent = float(top_percent)
        self.nn_chunk_size = int(nn_chunk_size)
        self.mask_patches = bool(mask_patches)
        self.memory_bank: np.ndarray | None = None

    def _filter_features(self, features: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
        if not self.mask_patches:
            return features
        mask = pca_patch_mask(features, grid_shape)
        if np.count_nonzero(mask) < 8:
            return features
        return features[mask]

    def fit(
        self,
        rois_bgr: list[np.ndarray],
        threshold_policy: dict[str, Any],
        augment_rotations: bool = False,
        max_memory_patches: int = 80000,
    ) -> dict[str, Any]:
        feats_list: list[np.ndarray] = []
        train_scores_rois: list[np.ndarray] = []
        rotations = [0, 1, 2, 3] if augment_rotations else [0]
        for roi in rois_bgr:
            for rot in rotations:
                src = np.rot90(roi, rot).copy() if rot else roi
                feats, grid = self.extractor.extract(src)
                feats = self._filter_features(feats, grid)
                feats_list.append(feats)
            train_scores_rois.append(roi)
        if not feats_list:
            raise RuntimeError("No DINO features extracted for the memory bank.")
        memory = np.concatenate(feats_list, axis=0).astype(np.float32)
        if max_memory_patches and memory.shape[0] > max_memory_patches:
            rng = np.random.default_rng(42)
            idx = rng.choice(memory.shape[0], size=max_memory_patches, replace=False)
            memory = memory[idx]
        self.memory_bank = _normalize_rows(memory)

        scores = np.array([self.score_roi(roi).score for roi in train_scores_rois], dtype=np.float32)
        percentile = float(threshold_policy.get("percentile", 99.5))
        std_factor = float(threshold_policy.get("std_factor", 3.0))
        min_threshold = float(threshold_policy.get("min_threshold", 0.0))
        self.threshold = max(
            float(np.percentile(scores, percentile)),
            float(np.mean(scores) + std_factor * np.std(scores)),
            min_threshold,
        )
        return {
            "threshold": self.threshold,
            "memory_patches": int(self.memory_bank.shape[0]),
            "feature_dim": int(self.memory_bank.shape[1]),
            "train_score_mean": float(np.mean(scores)),
            "train_score_std": float(np.std(scores)),
            "train_count": int(scores.size),
        }

    def _nn_distances(self, features: np.ndarray) -> np.ndarray:
        if self.memory_bank is None:
            raise RuntimeError("AnomalyDINO memory bank is empty.")
        best = np.full((features.shape[0],), -np.inf, dtype=np.float32)
        for start in range(0, self.memory_bank.shape[0], self.nn_chunk_size):
            chunk = self.memory_bank[start : start + self.nn_chunk_size]
            sim = features @ chunk.T
            best = np.maximum(best, np.max(sim, axis=1))
        return 1.0 - best

    def score_roi(self, roi_bgr: np.ndarray) -> DinoScore:
        features, grid = self.extractor.extract(roi_bgr)
        distances = self._nn_distances(features)
        k = max(1, int(np.ceil(distances.size * self.top_percent / 100.0)))
        top = np.partition(distances, -k)[-k:]
        score = float(np.mean(top))
        dist_img = distances.reshape(grid)
        heat = cv2.resize(dist_img, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        heat = cv2.GaussianBlur(heat, (0, 0), 4)
        heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        pred = "NG" if score > self.threshold else "OK"
        return DinoScore(score=score, pred=pred, distances=distances, heatmap=heat)

    def save(self, model_dir: Path, extra: dict[str, Any] | None = None) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Cannot save empty AnomalyDINO model.")
        model_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model_dir / "anomaly_dino_memory_bank.npz", memory_bank=self.memory_bank)
        meta = {
            "threshold": self.threshold,
            "top_percent": self.top_percent,
            "nn_chunk_size": self.nn_chunk_size,
            "mask_patches": self.mask_patches,
            "model_name": self.extractor.model_name,
            "weight_path": str(self.extractor.weight_path),
            "input_size": self.extractor.input_size,
        }
        if extra:
            meta.update(extra)
        save_json(meta, model_dir / "anomaly_dino_meta.json")

    @classmethod
    def load(cls, model_dir: Path, device: str = "auto") -> "AnomalyDinoModel":
        meta = load_json(model_dir / "anomaly_dino_meta.json")
        extractor = DinoV2Extractor(
            meta["model_name"],
            Path(meta["weight_path"]),
            input_size=int(meta["input_size"]),
            device=device,
        )
        obj = cls(
            extractor=extractor,
            threshold=float(meta["threshold"]),
            top_percent=float(meta.get("top_percent", 1.0)),
            nn_chunk_size=int(meta.get("nn_chunk_size", 8192)),
            mask_patches=bool(meta.get("mask_patches", False)),
        )
        data = np.load(model_dir / "anomaly_dino_memory_bank.npz")
        obj.memory_bank = data["memory_bank"].astype(np.float32)
        return obj
