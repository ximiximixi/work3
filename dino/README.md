# Industrial Cap Missing/Assembly Anomaly Detection

This project implements the big-assignment plan in the current folder:

- video preprocessing and ROI visualization
- a fast yellow-cap color baseline
- an optional AnomalyDINO backend using DINOv2 local weights
- prediction CSVs, metric evaluation, score plots, demo video rendering
- PDF report generation

## Environment

Use the bundled Codex Python runtime in this desktop session, because it already has OpenCV, NumPy, YAML, matplotlib, sklearn, Pillow, and reportlab:

```powershell
& "C:\Users\11816\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m dino_ad.cli run_all --config configs/video_a.yaml
```

PyTorch is optional. It is required only for:

```powershell
python -m dino_ad.cli build_memory_bank --config configs/video_a.yaml --method anomaly_dino
python -m dino_ad.cli infer_video --config configs/video_a.yaml --method anomaly_dino --write-heatmaps
```

The AnomalyDINO loader expects the local DINOv2 weight files already present in this folder, for example `dinov2_vits14_pretrain.pth`.

## Main Commands

```powershell
# 1. Extract sampled frames and ROI overlays
python -m dino_ad.cli extract_frames --config configs/video_a.yaml

# 2. Build the fast baseline model from train_range_sec only
python -m dino_ad.cli build_memory_bank --config configs/video_a.yaml --method baseline

# 3. Run inference on the test split
python -m dino_ad.cli infer_video --config configs/video_a.yaml --method baseline --write-heatmaps

# 4. Evaluate. If labels are missing, this writes labels_template.csv.
python -m dino_ad.cli evaluate --pred outputs/video_a/predictions/baseline/test_predictions.csv --out outputs/video_a/metrics/baseline_metrics.json

# 5. Render the 15-second demo video
python -m dino_ad.cli render_demo --config configs/video_a.yaml --pred outputs/video_a/predictions/baseline/test_predictions.csv

# Advanced realtime dashboard demo for the final two minutes at 8 fps
python -m dino_ad.cli render_demo --config configs/video_a.yaml --pred outputs/video_a/predictions/baseline/test_predictions.csv --advanced --last-minutes 2 --fps 8 --out outputs/video_a/demo_advanced_last2min_8fps.mp4

# 6. Generate PDF report
python -m dino_ad.cli generate_report --config configs/video_a.yaml --pred outputs/video_a/predictions/baseline/test_predictions.csv --metrics outputs/video_a/metrics/baseline_metrics.json
```

For a quick smoke test:

```powershell
python -m dino_ad.cli run_all --config configs/video_a.yaml --extract-max-frames 12 --train-max-frames 8 --infer-max-frames 20
```

## Outputs

Key artifacts are written under `outputs/video_a/`:

- `metadata/video_meta.json`
- `metadata/frames.csv`
- `figures/roi_contact_sheet.jpg`
- `figures/score_timeline.png`
- `models/color_baseline.json`
- `predictions/baseline/test_predictions.csv`
- `metrics/baseline_metrics.json`
- `metrics/labels_template.csv` if no labels are provided
- `demo_15s.mp4`
- `report.pdf`

## Labels

To compute accuracy, precision, recall, F1, and false positive rate:

1. Run `evaluate` once without labels.
2. Fill the generated `labels_template.csv` with `OK` or `NG`.
3. Rerun:

```powershell
python -m dino_ad.cli evaluate --pred outputs/video_a/predictions/baseline/test_predictions.csv --labels outputs/video_a/metrics/labels_template.csv --out outputs/video_a/metrics/baseline_metrics.json
```

Training and threshold selection use only `train_range_sec`; labels are used only for evaluation.
