from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import output_dir
from .video import load_json, read_rows_csv


def _register_cjk_font() -> str:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("CJKFont", path))
                return "CJKFont"
            except Exception:
                pass
    return "Helvetica"


def _style():
    font = _register_cjk_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle", fontName=font, fontSize=20, leading=28, spaceAfter=16))
    styles.add(ParagraphStyle(name="CNHeading", fontName=font, fontSize=14, leading=20, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="CNBody", fontName=font, fontSize=10.5, leading=16, spaceAfter=6))
    return styles, font


def build_report(cfg: dict, pred_csv: Path, metrics_json: Path | None = None, out_pdf: Path | None = None) -> dict:
    out_dir = output_dir(cfg)
    out_pdf = out_pdf or (out_dir / "report.pdf")
    styles, font = _style()
    story = []
    story.append(Paragraph("基于 AnomalyDINO 的工业装配件缺失检测实验报告", styles["CNTitle"]))
    story.append(
        Paragraph(
            "本报告对应大作业03：仅使用视频前段正常装配样本建立正常参考模型，"
            "在后段视频中识别盖子缺失或装配异常，并输出逐帧 OK/NG。",
            styles["CNBody"],
        )
    )

    meta_path = out_dir / "metadata" / "video_meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        rows = [
            ["视频", Path(meta["video_path"]).name],
            ["分辨率", f'{meta["width"]} x {meta["height"]}'],
            ["帧率/帧数", f'{meta["fps"]:.3f} fps / {meta["frame_count"]} frames'],
            ["时长", f'{meta["duration_sec"]:.2f} s'],
            ["ROI", str(meta["roi_xyxy"])],
        ]
        story.append(Paragraph("1. 数据与预处理", styles["CNHeading"]))
        story.append(_table(rows, font))
        story.append(
            Paragraph(
                "预处理阶段按配置抽取关键帧，并使用帧差与 Laplacian 清晰度过滤运动过程中的模糊帧。"
                "ROI 采用手工框选，保证透明圆筒与黄色盖子区域被完整覆盖。",
                styles["CNBody"],
            )
        )

    story.append(Paragraph("2. 方法", styles["CNHeading"]))
    story.append(
        Paragraph(
            "轻量基线使用 HSV 颜色阈值统计 ROI 中黄色/橙色盖子区域占比，盖子颜色面积显著低于训练段正常中位数时判为 NG。"
            "主方法 AnomalyDINO 使用 DINOv2 patch token 构建正常 memory bank，测试时计算每个 patch 到正常库的最近邻 cosine 距离，"
            "并采用论文推荐的 top 1% patch 距离均值作为图像级异常分数。",
            styles["CNBody"],
        )
    )

    story.append(Paragraph("3. 实验结果", styles["CNHeading"]))
    pred_rows = read_rows_csv(pred_csv) if pred_csv.exists() else []
    if pred_rows:
        ng_count = sum(1 for r in pred_rows if r["pred"].upper() == "NG")
        lat = [float(r["latency_ms"]) for r in pred_rows if r.get("latency_ms")]
        rows = [
            ["预测文件", str(pred_csv)],
            ["样本数", str(len(pred_rows))],
            ["NG 数", str(ng_count)],
            ["平均延迟", f"{sum(lat) / len(lat):.2f} ms" if lat else "N/A"],
            ["估计 FPS", f"{1000.0 / (sum(lat) / len(lat)):.2f}" if lat else "N/A"],
        ]
        story.append(_table(rows, font))
    if metrics_json and metrics_json.exists():
        metrics = load_json(metrics_json)
        if metrics.get("status") == "ok":
            rows = [
                ["Accuracy", f'{metrics["accuracy"]:.4f}'],
                ["Precision", f'{metrics["precision"]:.4f}'],
                ["Recall", f'{metrics["recall"]:.4f}'],
                ["F1", f'{metrics["f1"]:.4f}'],
                ["False Positive Rate", f'{metrics["false_positive_rate"]:.4f}'],
            ]
            story.append(_table(rows, font))
        else:
            story.append(Paragraph("当前未提供人工 OK/NG 标签，因此报告只给出预测统计和标签模板。", styles["CNBody"]))

    score_fig = out_dir / "figures" / "score_timeline.png"
    if score_fig.exists():
        story.append(Image(str(score_fig), width=16 * cm, height=5.8 * cm))
        story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("4. 分析与改进", styles["CNHeading"]))
    story.append(
        Paragraph(
            "无监督方案的优势是无需收集大量 NG 缺陷样本，也不需要逐帧画框标注；相比 YOLO 类监督检测，"
            "更适合单条产线、小样本和快速验证场景。主要风险包括：ROI 未覆盖目标、训练段正常样本多样性不足、"
            "光照变化导致颜色基线误报、以及语义型异常不一定带来明显 patch 外观差异。后续可补充第二段视频、"
            "尝试 DINOv2-B/DINOv2-reg、加入 PCA mask 或更细的多目标 ROI。",
            styles["CNBody"],
        )
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm)
    doc.build(story)
    return {"report_pdf": str(out_pdf)}


def _table(rows: list[list[str]], font: str) -> Table:
    table = Table(rows, colWidths=[4.0 * cm, 11.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#edf2f7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table
