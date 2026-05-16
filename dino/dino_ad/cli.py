from __future__ import annotations

import argparse
from pathlib import Path

from .anomaly_dino import MissingTorchError
from .config import load_config, output_dir
from .metrics import evaluate_predictions
from .pipeline import build_memory_bank, infer_video
from .preprocess import extract_frames
from .report import build_report
from .visualize import make_figures_for_predictions, render_demo


def cmd_extract(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    result = extract_frames(cfg, split=args.split, max_frames=args.max_frames, save_images=not args.no_images)
    print(result)


def cmd_build(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    result = build_memory_bank(cfg, method=args.method, max_frames=args.max_frames)
    print(result)


def cmd_infer(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    result = infer_video(
        cfg,
        method=args.method,
        split=args.split,
        max_frames=args.max_frames,
        write_heatmaps=args.write_heatmaps,
    )
    pred_csv = Path(result["prediction_csv"])
    fig_dir = output_dir(cfg) / "figures"
    make_figures_for_predictions(pred_csv, fig_dir)
    print(result)


def cmd_evaluate(args: argparse.Namespace) -> None:
    result = evaluate_predictions(Path(args.pred), Path(args.labels) if args.labels else None, Path(args.out))
    print(result)


def cmd_render_demo(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    result = render_demo(
        cfg,
        Path(args.pred),
        out_video=Path(args.out) if args.out else None,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        fps=args.fps,
        advanced=args.advanced,
        last_minutes=args.last_minutes,
    )
    print(result)


def cmd_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    result = build_report(
        cfg,
        Path(args.pred),
        metrics_json=Path(args.metrics) if args.metrics else None,
        out_pdf=Path(args.out) if args.out else None,
    )
    print(result)


def cmd_run_all(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print("[1/6] Extracting frames")
    print(extract_frames(cfg, split="all", max_frames=args.extract_max_frames, save_images=True))
    print("[2/6] Building baseline model")
    print(build_memory_bank(cfg, method="baseline", max_frames=args.train_max_frames))
    print("[3/6] Inferring test split")
    infer_result = infer_video(
        cfg,
        method="baseline",
        split="test",
        max_frames=args.infer_max_frames,
        write_heatmaps=True,
    )
    pred_csv = Path(infer_result["prediction_csv"])
    print(infer_result)
    print("[4/6] Making figures")
    print(make_figures_for_predictions(pred_csv, output_dir(cfg) / "figures"))
    print("[5/6] Evaluating or writing label template")
    metrics_path = output_dir(cfg) / "metrics" / "baseline_metrics.json"
    print(evaluate_predictions(pred_csv, Path(args.labels) if args.labels else None, metrics_path))
    print("[6/6] Rendering demo and report")
    print(render_demo(cfg, pred_csv))
    print(build_report(cfg, pred_csv, metrics_json=metrics_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial cap anomaly detection toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract_frames")
    p.add_argument("--config", default="configs/video_a.yaml")
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--max-frames", type=int)
    p.add_argument("--no-images", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("build_memory_bank")
    p.add_argument("--config", default="configs/video_a.yaml")
    p.add_argument("--method", default="baseline", choices=["baseline", "anomaly_dino"])
    p.add_argument("--max-frames", type=int)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("infer_video")
    p.add_argument("--config", default="configs/video_a.yaml")
    p.add_argument("--method", default="baseline", choices=["baseline", "anomaly_dino"])
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--max-frames", type=int)
    p.add_argument("--write-heatmaps", action="store_true")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("evaluate")
    p.add_argument("--pred", required=True)
    p.add_argument("--labels")
    p.add_argument("--out", default="outputs/video_a/metrics/metrics.json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("render_demo")
    p.add_argument("--config", default="configs/video_a.yaml")
    p.add_argument("--pred", required=True)
    p.add_argument("--out")
    p.add_argument("--start-sec", type=float)
    p.add_argument("--duration-sec", type=float)
    p.add_argument("--fps", type=float)
    p.add_argument("--advanced", action="store_true")
    p.add_argument("--last-minutes", type=float)
    p.set_defaults(func=cmd_render_demo)

    p = sub.add_parser("generate_report")
    p.add_argument("--config", default="configs/video_a.yaml")
    p.add_argument("--pred", required=True)
    p.add_argument("--metrics")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run_all")
    p.add_argument("--config", default="configs/video_a.yaml")
    p.add_argument("--extract-max-frames", type=int)
    p.add_argument("--train-max-frames", type=int)
    p.add_argument("--infer-max-frames", type=int)
    p.add_argument("--labels")
    p.set_defaults(func=cmd_run_all)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except MissingTorchError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
