# Industrial Video Defect Detection

This repository contains a computer-vision project for industrial video defect detection. The final task is to detect whether the beige side sleeve/cap is missing in tiny side-interface regions of a target sample.

GitHub: https://github.com/ximiximixi/work3

## Demo Videos

The README shows animated previews of the two final demonstration videos prepared on May 16, with MP4 links kept for full playback.

### Demo 1

Source video: `5月16日.mp4`

![Demo 1](docs/media/may16_demo_01.gif)

[Open MP4 demo 1](docs/media/may16_demo_01.mp4)

### Demo 2

Source video: `5月16日(1).mp4`

![Demo 2](docs/media/may16_demo_02.gif)

[Open MP4 demo 2](docs/media/may16_demo_02.mp4)

## Main Pipelines

- `defect_detection_pipeline.py`
  - Early coarse-ROI / PatchCore-style anomaly detector for `20241031_222226.mp4`.
  - Uses circle/gate localization, handcrafted patch features, a normal memory bank, nearest-neighbor anomaly scoring, hybrid color/texture boosting, event filtering, and diagnostic videos.

- `cap_defect_detection_pipeline.py`
  - Intermediate cap/large-ROI detector for `20241101_161258.mp4`.
  - Provides blue-cap anchor localization reused by the final micro-ROI detector.

- `prepare_cap_annotation_set.py`
  - Extracts right-half frames for ROI calibration.

- `cap_annotation_tool/`
  - Lightweight browser ROI calibration tool.
  - Saves coordinates in original video coordinates by adding `offset_x=960`.

- `side_micro_roi_detector.py`
  - Final fixed side micro-ROI detector.
  - Trains an ExtraTrees `PRESENT/MISSING/BACKGROUND` patch classifier from ROI calibration samples.
  - Uses fixed upper/lower horizontal bands, blue-cap constrained candidates, beige sleeve evidence rules, and event aggregation.

- `make_side_micro_showcase_video.py`
  - Renders the final 60-second 8 FPS showcase video with a diagnostic dashboard.

## Key Commands

```powershell
python defect_detection_pipeline.py run ^
  --video 20241031_222226.mp4 ^
  --out defect_outputs ^
  --train-end 350 ^
  --val-start 350 ^
  --val-end 360 ^
  --scan-start 360 ^
  --scan-step 0.5 ^
  --preview-start 360 ^
  --preview-end 430

python defect_detection_pipeline.py review-video ^
  --video 20241031_222226.mp4 ^
  --observations defect_outputs/observations.csv ^
  --model-dir defect_outputs/model ^
  --out defect_outputs/visuals/07_sampled_2fps_review.mp4 ^
  --fps 2 ^
  --sample-step 0.5 ^
  --snapshots-dir defect_outputs/visuals/07_review_snapshots

python prepare_cap_annotation_set.py

python side_micro_roi_detector.py run ^
  --video 20241101_161258.mp4 ^
  --annotations cap_annotation_tool ^
  --out side_micro_outputs ^
  --sample-step 0.5

python make_side_micro_showcase_video.py ^
  --video 20241101_161258.mp4 ^
  --annotations cap_annotation_tool ^
  --out side_micro_outputs/visuals/03_last60s_8fps_showcase.mp4 ^
  --seconds 60 ^
  --fps 8
```

## Report

The LaTeX report draft is:

- `paper/cv_defect_detection_report.tex`

It covers:

- the complete `20241031_222226.mp4` coarse-ROI / PatchCore-style experiment;
- the failed automatic large-ROI attempts;
- side micro-ROI calibration;
- final fixed upper/lower band micro-ROI classification;
- event statistics and visualization videos;
- limitations, missing ground truth, and future improvements.

## Notes on Accuracy

The current project outputs detection statistics and internal holdout metrics, but it does not yet include full frame-level or event-level ground truth for both videos. Strict per-video accuracy and multi-video mean accuracy should therefore be computed only after adding ground-truth labels.
