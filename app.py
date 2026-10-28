"""
Gradio dashboard for structural damage detection.
Loads weights/best.pt and results/*.json produced by colab.ipynb.
Never trains anything.

Run via:
    uv run python main.py app
"""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go
from ultralytics import YOLO

# ── Load artifacts ──────────────────────────────────────────────────────────
WEIGHTS_PATH = Path("weights/best.pt")
RESULTS_DIR  = Path("results")


def _load_json(name):
    path = RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


model      = YOLO(str(WEIGHTS_PATH)) if WEIGHTS_PATH.exists() else None
metrics    = _load_json("metrics.json")
cost_sweep = _load_json("cost_sweep.json")
op_point   = _load_json("operating_point.json")

COST_FN   = op_point["cost_fn"]   if op_point else 150_000.0
COST_FP   = op_point["cost_fp"]   if op_point else  50_000.0
BASE_RATE = op_point["base_rate"] if op_point else 0.15


# ── Dataset discovery ─────────────────────────────────────────────────────────
DATASET_DIR = None
for _c in ["detectorn2-1", "detectorn2-v1-1", "detectorn2_1", "detectorn2-vh2zv-1", "sample_data"]:
    if Path(_c).exists():
        DATASET_DIR = Path(_c)
        break


# ── Class helpers ─────────────────────────────────────────────────────────────
_CLASS_COLORS = {
    "total_damage": (220,  50,  50),
    "major_damage": (220, 120,  20),
    "minor_damage": (220, 200,  20),
    "no_damage":    ( 50, 200,  50),
    "gdfg":         (160, 160, 160),
}
_BAR_COLORS = {
    "total_damage": "#e74c3c",
    "major_damage": "#e67e22",
    "minor_damage": "#f1c40f",
    "no_damage":    "#2ecc71",
    "gdfg":         "#95a5a6",
}


def _class_names() -> list[str]:
    if metrics and "finetuned" in metrics:
        return list(metrics["finetuned"]["per_class"].keys())
    return []


def _occ_idx() -> int:
    for i, name in enumerate(_class_names()):
        if "total_damage" in name.lower():
            return i
    return 0


# ── Cost helpers ──────────────────────────────────────────────────────────────
def _interp_sweep(threshold):
    if not cost_sweep:
        return 0.0, 0.0
    ts = [x["threshold"] for x in cost_sweep]
    ps = [x["precision"] for x in cost_sweep]
    rs = [x["recall"]    for x in cost_sweep]
    return float(np.interp(threshold, ts, ps)), float(np.interp(threshold, ts, rs))


def _cost_at(threshold):
    p, r = _interp_sweep(threshold)
    e_fn = 1000 * BASE_RATE * (1 - r)
    e_fp = (1000 * BASE_RATE * r * (1 - p) / p) if p > 0 else 1000 * (1 - BASE_RATE)
    return COST_FN * e_fn + COST_FP * e_fp


def _recompute_optimal():
    if not cost_sweep:
        return op_point["threshold"] if op_point else 0.5
    best_t, best_c = 0.5, float("inf")
    for row in cost_sweep:
        p, r = row["precision"], row["recall"]
        e_fn = 1000 * BASE_RATE * (1 - r)
        e_fp = (1000 * BASE_RATE * r * (1 - p) / p) if p > 0 else 1000 * (1 - BASE_RATE)
        c = COST_FN * e_fn + COST_FP * e_fp
        if c < best_c:
            best_c, best_t = c, row["threshold"]
    return best_t


OPTIMAL_T = _recompute_optimal()


# ── Val image helpers ─────────────────────────────────────────────────────────
def _build_val_map() -> dict[str, str]:
    if not DATASET_DIR:
        return {}
    for sub in ["valid/images", "val/images"]:
        folder = DATASET_DIR / sub
        if folder.exists():
            return {p.stem[:50]: str(p) for p in sorted(folder.glob("*.[jp][pn]g"))}
    return {}


VAL_MAP = _build_val_map()


def _build_test_examples() -> list[list[str]]:
    if not DATASET_DIR:
        return []
    test_dir = DATASET_DIR / "test" / "images"
    if not test_dir.exists():
        return []
    paths = sorted(test_dir.glob("*.[jp][pn]g"))[:5]
    return [[str(p)] for p in paths]


TEST_EXAMPLES = _build_test_examples()

_DEFAULT_IMG = next(
    (k for k in VAL_MAP if "DJI_20231109095923_0051" in k),
    list(VAL_MAP.keys())[0] if VAL_MAP else None,
)


def _draw_gt_boxes(image_path: str) -> np.ndarray:
    path       = Path(image_path)
    label_path = path.parent.parent / "labels" / (path.stem + ".txt")
    img        = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    H, W       = img.shape[:2]
    names      = _class_names()
    if not label_path.exists():
        return img
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        name  = names[cls] if cls < len(names) else str(cls)
        color = _CLASS_COLORS.get(name, (100, 100, 220))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return img


def _outcome_label(image_path: str, t_low: float, t_high: float) -> str:
    """Return pass / human review / fail for a single image."""
    if model is None:
        return ""
    res   = model.predict(image_path, conf=0.01, verbose=False)
    boxes = res[0].boxes
    occ   = _occ_idx()
    if boxes is None or len(boxes) == 0:
        return "✅ **PASS** — no damage detected"
    confs = [float(c) for c, cls in zip(boxes.conf.tolist(), boxes.cls.tolist())
             if int(cls) == occ]
    if not confs:
        return "✅ **PASS** — no total_damage detected"
    max_conf = max(confs)
    if max_conf >= t_high:
        return f"🔴 **FAIL** — total_damage detected (conf {max_conf:.2f} ≥ {t_high:.2f})"
    if max_conf >= t_low:
        return f"🟡 **HUMAN REVIEW** — uncertain (conf {max_conf:.2f} between {t_low:.2f}–{t_high:.2f})"
    return "✅ **PASS** — confidence below pass threshold"


# ── Tab 1 — Upload & Predict ──────────────────────────────────────────────────
def predict_upload(image_path, t_low, t_high):
    if model is None:
        return None, "No weights found — run training first."
    if image_path is None:
        return None, "Upload an image to see predictions."
    t = float(t_high)
    results  = model.predict(image_path, conf=t, verbose=False)
    pred_img = results[0].plot()
    outcome  = _outcome_label(image_path, float(t_low), float(t_high))
    return pred_img, outcome


# ── Tab 2 — Predict vs Ground Truth ──────────────────────────────────────────
def predict_vs_gt(display_label, threshold):
    if model is None:
        return None, None, "No weights found — run training first."
    path = VAL_MAP.get(display_label)
    if not path:
        return None, None, "Select an image."

    t        = float(threshold)
    results  = model.predict(path, conf=t, verbose=False)
    pred_img = results[0].plot()
    gt_img   = _draw_gt_boxes(path)

    boxes   = results[0].boxes
    occ_idx = _occ_idx()

    n_total_pred = int((boxes.cls == occ_idx).sum()) if boxes is not None else 0
    n_other_pred = (len(boxes) - n_total_pred) if boxes is not None else 0

    label_path = Path(path).parent.parent / "labels" / (Path(path).stem + ".txt")
    n_total_gt = n_other_gt = 0
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) >= 1:
                if int(parts[0]) == occ_idx:
                    n_total_gt += 1
                else:
                    n_other_gt += 1

    missed = max(0, n_total_gt   - n_total_pred)
    extra  = max(0, n_total_pred - n_total_gt)

    summary = (
        f"**Ground truth:** {n_total_gt} total_damage · {n_other_gt} other &nbsp;|&nbsp; "
        f"**Model ({t:.2f}):** {n_total_pred} total_damage · {n_other_pred} other &nbsp;|&nbsp; "
        f"**Missed (FN):** {missed} &nbsp;|&nbsp; **False alarms (FP):** {extra}"
    )
    return pred_img, gt_img, summary


# ── Tab 3 — Decision Thresholds ───────────────────────────────────────────────
def decision_impact(t_low, t_high):
    t_low  = float(t_low)
    t_high = float(t_high)
    if t_low >= t_high:
        t_low = max(0.01, t_high - 0.05)

    p_low,  r_low  = _interp_sweep(t_low)
    p_high, r_high = _interp_sweep(t_high)

    n_pos = 1000 * BASE_RATE    # ~150 total_damage
    n_neg = 1000 * (1 - BASE_RATE)  # ~850 other

    # Automated FAIL (above t_high)
    tp_fail = n_pos * r_high
    fp_fail = min((tp_fail * (1 - p_high) / p_high) if p_high > 0 else n_neg, n_neg)

    # Flagged at t_low (review + fail zone)
    tp_low = n_pos * r_low
    fp_low = min((tp_low * (1 - p_low) / p_low) if p_low > 0 else n_neg, n_neg)

    # Automated PASS (below t_low)
    fn_pass = n_pos - tp_low          # true positives missed entirely
    tn_pass = n_neg - fp_low          # correctly passed

    # Human REVIEW zone
    tp_review = tp_low  - tp_fail
    fp_review = fp_low  - fp_fail
    n_review  = tp_review + fp_review

    # Costs: only automated decisions carry cost; reviewers are assumed correct
    cost_fn_auto = fn_pass * COST_FN
    cost_fp_auto = fp_fail * COST_FP
    total_cost   = cost_fn_auto + cost_fp_auto

    review_pct  = n_review / 1000 * 100
    fail_pct    = (tp_fail + fp_fail) / 1000 * 100
    pass_pct    = 100 - review_pct - fail_pct

    # Confusion matrix (automated decisions only — review excluded)
    cm_text = [
        [f"{int(tp_fail)} correctly failed",
         f"{int(tp_review)} → human review",
         f"{int(fn_pass)} missed<br><b>−€{cost_fn_auto:,.0f}</b>"],
        [f"{int(fp_fail)} wrongly failed<br><b>−€{cost_fp_auto:,.0f}</b>",
         f"{int(fp_review)} → human review",
         f"{int(tn_pass)} correctly passed"],
    ]
    cm_z = [[0, 0, cost_fn_auto], [cost_fp_auto, 0, 0]]

    fig = go.Figure(go.Heatmap(
        z=cm_z,
        x=["Auto: FAIL", "Human review", "Auto: PASS"],
        y=["Actually: total_damage", "Actually: other"],
        colorscale=[[0, "#e8f5e9"], [0.01, "#fff9c4"], [1, "#c62828"]],
        zmin=0, zmax=max(cost_fn_auto, cost_fp_auto, 1),
        text=cm_text, texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(
        title=(f"Per 1,000 inspections — pass &lt;{t_low:.2f} · review {t_low:.2f}–{t_high:.2f} · fail ≥{t_high:.2f}"
               f"  |  Total automated cost: €{total_cost:,.0f}"),
        margin=dict(t=70, b=20, l=20, r=20), height=320, font=dict(size=12),
    )

    summary = (
        f"🟢 **Pass:** {pass_pct:.1f}% &nbsp;|&nbsp; "
        f"🟡 **Review:** {review_pct:.1f}% ({int(n_review)} cases/1k) &nbsp;|&nbsp; "
        f"🔴 **Fail:** {fail_pct:.1f}% &nbsp;|&nbsp; "
        f"Automated cost: **−€{total_cost:,.0f} / 1k** "
        f"(FN: −€{cost_fn_auto:,.0f} · FP: −€{cost_fp_auto:,.0f})"
    )
    return fig, summary


# ── Tab 4 — Cost Curve & Per-Class ───────────────────────────────────────────
def cost_curve_plot():
    if not cost_sweep:
        fig = go.Figure()
        fig.add_annotation(text="Run training first.", showarrow=False, font_size=16)
        return fig
    ts       = [x["threshold"] for x in cost_sweep]
    costs    = [_cost_at(t) for t in ts]
    opt_cost = _cost_at(OPTIMAL_T)

    worst_cost = COST_FN * (1000 * BASE_RATE) + COST_FP * (1000 * (1 - BASE_RATE))

    coco_p = metrics["zero_shot_coco"]["precision"] if metrics else 0.0
    coco_r = metrics["zero_shot_coco"]["recall"]    if metrics else 0.0
    e_fn_coco = 1000 * BASE_RATE * (1 - coco_r)
    e_fp_coco = (1000 * BASE_RATE * coco_r * (1 - coco_p) / coco_p) if coco_p > 0 else 1000 * (1 - BASE_RATE)
    coco_cost = COST_FN * e_fn_coco + COST_FP * e_fp_coco

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[ts[0], ts[-1]], y=[worst_cost, worst_cost],
                             mode="lines", line=dict(color="#7f7f7f", width=1.5, dash="dot"),
                             name=f"Worst case (€{worst_cost:,.0f})"))
    fig.add_trace(go.Scatter(x=[ts[0], ts[-1]], y=[coco_cost, coco_cost],
                             mode="lines", line=dict(color="#ff7f0e", width=1.5, dash="dash"),
                             name=f"COCO zero-shot (€{coco_cost:,.0f})"))
    fig.add_trace(go.Scatter(x=ts, y=costs, mode="lines",
                             line=dict(color="#1f77b4", width=2.5),
                             name="Finetuned model"))
    fig.add_vline(x=OPTIMAL_T, line_dash="dash", line_color="crimson",
                  annotation_text=f"Optimal {OPTIMAL_T:.2f}",
                  annotation_position="top right")
    fig.add_trace(go.Scatter(x=[OPTIMAL_T], y=[opt_cost], mode="markers",
                             marker=dict(color="crimson", size=10, symbol="diamond"),
                             name=f"Optimal (€{opt_cost:,.0f})"))
    fig.update_layout(
        title="Expected Inspection Cost vs Confidence Threshold (per 1,000 cases)",
        xaxis_title="Confidence threshold", yaxis_title="Cost (€)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=50, b=40, l=60, r=20), height=420,
    )
    return fig


def per_class_html():
    if not metrics or "finetuned" not in metrics:
        return "<p>Run training first to see per-class results.</p>"
    ft      = metrics["finetuned"]
    coco    = metrics.get("zero_shot_coco", {})
    trivial = metrics.get("trivial", {})
    rows = "".join(
        f"<tr><td style='padding:6px 12px'>{cls}</td>"
        f"<td style='padding:6px 12px;text-align:center'>{v['AP50']:.3f}</td>"
        f"<td style='padding:6px 12px;text-align:center'>{v['precision']:.3f}</td>"
        f"<td style='padding:6px 12px;text-align:center'>{v['recall']:.3f}</td></tr>"
        for cls, v in ft["per_class"].items()
    )
    return f"""
    <h4>Per-class results — finetuned yolo11n</h4>
    <table style='border-collapse:collapse;font-size:13px'>
      <thead><tr style='background:#f0f0f0'>
        <th style='padding:6px 12px'>Class</th>
        <th style='padding:6px 12px'>AP50</th>
        <th style='padding:6px 12px'>Precision</th>
        <th style='padding:6px 12px'>Recall</th>
      </tr></thead><tbody>{rows}</tbody></table>
    <h4 style='margin-top:16px'>Baseline comparison</h4>
    <table style='border-collapse:collapse;font-size:13px'>
      <thead><tr style='background:#f0f0f0'>
        <th style='padding:6px 12px'>Model</th>
        <th style='padding:6px 12px'>mAP50</th>
        <th style='padding:6px 12px'>mAP50-95</th>
        <th style='padding:6px 12px'>Precision</th>
        <th style='padding:6px 12px'>Recall</th>
      </tr></thead><tbody>
        <tr><td style='padding:6px 12px'>Finetuned yolo11n</td>
            <td style='padding:6px 12px;text-align:center'>{ft['mAP50']:.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{ft['mAP50-95']:.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{ft['precision']:.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{ft['recall']:.3f}</td></tr>
        <tr><td style='padding:6px 12px'>Zero-shot COCO (any detection → total_damage)</td>
            <td style='padding:6px 12px;text-align:center'>{coco.get('mAP50', 0):.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{coco.get('mAP50-95', 0):.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{coco.get('precision', 0):.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{coco.get('recall', 0):.3f}</td></tr>
        <tr><td style='padding:6px 12px'>Trivial (predict all as total_damage)</td>
            <td style='padding:6px 12px;text-align:center'>{trivial.get('mAP50', 0):.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{trivial.get('mAP50-95', 0):.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{trivial.get('precision', 0):.3f}</td>
            <td style='padding:6px 12px;text-align:center'>{trivial.get('recall', 0):.3f}</td></tr>
      </tbody></table>"""


# ── Tab 5 — Batch Inspector ───────────────────────────────────────────────────
def run_batch(threshold):
    if model is None:
        return None, "No weights found — run training first."
    if not DATASET_DIR:
        return None, "Dataset not found locally."

    img_dir = None
    for sub in ["valid/images", "val/images"]:
        d = DATASET_DIR / sub
        if d.exists():
            img_dir = d
            break
    if not img_dir:
        return None, "Val images not found."

    t        = float(threshold)
    occ_idx  = _occ_idx()
    names    = _class_names()
    img_paths = sorted(img_dir.glob("*.[jp][pn]g"))

    img_class_flags = defaultdict(int)
    n_flagged_total = 0

    for img_path in img_paths:
        res   = model.predict(str(img_path), conf=t, verbose=False)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        classes_in_img = {names[int(cls)] if int(cls) < len(names) else str(int(cls))
                          for cls in boxes.cls.tolist()}
        for name in classes_in_img:
            img_class_flags[name] += 1
        if occ_idx in [int(c) for c in boxes.cls.tolist()]:
            n_flagged_total += 1

    n_images   = len(img_paths)
    p, r       = _interp_sweep(t)
    e_fn       = n_images * BASE_RATE * (1 - r)
    e_fp       = (n_images * BASE_RATE * r * (1 - p) / p) if p > 0 else n_images * (1 - BASE_RATE)
    batch_cost = COST_FN * e_fn + COST_FP * e_fp

    bar_colors = [_BAR_COLORS.get(k, "#7f8c8d") for k in img_class_flags]
    fig = go.Figure(go.Bar(
        x=list(img_class_flags.keys()),
        y=list(img_class_flags.values()),
        marker_color=bar_colors,
    ))
    fig.update_layout(
        title=f"Images flagged per class — {n_images} val images at conf ≥ {t:.2f}",
        xaxis_title="Damage class", yaxis_title="Images flagged (≥1 detection)",
        margin=dict(t=50, b=40, l=60, r=20), height=380,
    )
    summary = (
        f"**{n_images} images scanned** &nbsp;|&nbsp; "
        f"**{n_flagged_total} flagged as total_damage** &nbsp;|&nbsp; "
        f"Estimated batch cost: **€{batch_cost:,.0f}**"
    )
    return fig, summary


# ── Build dashboard ───────────────────────────────────────────────────────────
with gr.Blocks(title="Damage Detection Dashboard") as demo:
    gr.Markdown(
        "# Structural Damage Detection Dashboard\n"
        f"Optimal threshold: **{OPTIMAL_T:.2f}** &nbsp;|&nbsp; "
        f"Miss total_damage (FN): **€{COST_FN:,.0f}** &nbsp;|&nbsp; "
        f"False alarm (FP): **€{COST_FP:,.0f}** &nbsp;|&nbsp; "
        f"Total_damage base rate: **{BASE_RATE*100:.0f}%**"
    )

    # ── Tab 1: Upload & Predict ─────────────────────────────────────────────
    with gr.Tab("Upload & Predict"):
        gr.Markdown(
            "Upload any image to run the model. The pass/review/fail decision uses "
            "the two thresholds set in **Decision Thresholds**."
        )
        with gr.Row():
            t_low_ref  = gr.Number(value=round(OPTIMAL_T * 0.6, 2), visible=False)
            t_high_ref = gr.Number(value=OPTIMAL_T, visible=False)
        img_upload = gr.Image(type="filepath", label="Upload an image")
        if TEST_EXAMPLES:
            gr.Examples(
                examples=TEST_EXAMPLES,
                inputs=[img_upload],
                label="Sample test images (click to load)",
            )
        with gr.Row():
            up_t_low  = gr.Slider(0.01, 0.99, value=round(OPTIMAL_T * 0.6, 2),
                                  step=0.01, label="Pass threshold (below → pass)")
            up_t_high = gr.Slider(0.01, 0.99, value=OPTIMAL_T,
                                  step=0.01, label="Fail threshold (above → fail)")
        up_outcome = gr.Markdown()
        up_pred    = gr.Image(type="numpy", label="Model prediction")
        img_upload.change(predict_upload, [img_upload, up_t_low, up_t_high], [up_pred, up_outcome])
        up_t_low.change(predict_upload,  [img_upload, up_t_low, up_t_high], [up_pred, up_outcome])
        up_t_high.change(predict_upload, [img_upload, up_t_low, up_t_high], [up_pred, up_outcome])

    # ── Tab 2: Predict vs Ground Truth ─────────────────────────────────────
    with gr.Tab("Predict vs Ground Truth"):
        gr.Markdown(
            "Select a validation image. **Left** = model prediction. "
            "**Right** = ground truth labels. "
            "Shows missed `total_damage` (FN) and false alarms (FP)."
        )
        with gr.Row():
            img_select = gr.Dropdown(
                choices=list(VAL_MAP.keys()),
                value=_DEFAULT_IMG,
                label="Validation image", scale=3,
            )
            conf_sl = gr.Slider(0.01, 0.99, value=OPTIMAL_T, step=0.01,
                                label="Confidence threshold", scale=1)
        pred_md = gr.Markdown()
        with gr.Row():
            img_pred = gr.Image(type="numpy", label="Model prediction")
            img_gt   = gr.Image(type="numpy", label="Ground truth")
        img_select.change(predict_vs_gt, [img_select, conf_sl], [img_pred, img_gt, pred_md])
        conf_sl.change(predict_vs_gt,   [img_select, conf_sl], [img_pred, img_gt, pred_md])
        demo.load(lambda: predict_vs_gt(_DEFAULT_IMG, OPTIMAL_T), [], [img_pred, img_gt, pred_md])

    # ── Tab 3: Decision Thresholds ──────────────────────────────────────────
    with gr.Tab("Decision Thresholds"):
        gr.Markdown(
            "Set two thresholds to create **three outcomes** for each inspection:\n\n"
            "- 🟢 **Pass** — model confidence below pass threshold → no action needed\n"
            "- 🟡 **Human review** — confidence in the middle zone → send to inspector\n"
            "- 🔴 **Fail** — model confidence above fail threshold → dispatch response team\n\n"
            "The confusion matrix and cost update as you move the thresholds. "
            "Human review cases are assumed to be resolved correctly (zero automated cost)."
        )
        with gr.Row():
            dt_low  = gr.Slider(0.01, 0.98, value=round(OPTIMAL_T * 0.6, 2), step=0.01,
                                label="🟢 Pass threshold — below this → PASS")
            dt_high = gr.Slider(0.02, 0.99, value=OPTIMAL_T, step=0.01,
                                label="🔴 Fail threshold — above this → FAIL")
        dt_md  = gr.Markdown()
        dt_cm  = gr.Plot()
        dt_low.change(decision_impact,  [dt_low, dt_high], [dt_cm, dt_md])
        dt_high.change(decision_impact, [dt_low, dt_high], [dt_cm, dt_md])
        demo.load(lambda: decision_impact(round(OPTIMAL_T * 0.6, 2), OPTIMAL_T), [], [dt_cm, dt_md])

    # ── Tab 4: Cost Curve & Per-Class ───────────────────────────────────────
    with gr.Tab("Cost Curve & Per-Class"):
        gr.Markdown(
            "Expected inspection cost vs confidence threshold. "
            "The optimal threshold (red) minimises total cost. "
            "Baselines shown for comparison. Per-class metrics below."
        )
        curve_plt  = gr.Plot()
        class_html = gr.HTML()
        demo.load(lambda: (cost_curve_plot(), per_class_html()), [], [curve_plt, class_html])

    # ── Tab 5: Batch Inspector ──────────────────────────────────────────────
    with gr.Tab("Batch Inspector"):
        gr.Markdown(
            "Runs inference on all validation images at the selected threshold. "
            "Shows images flagged per damage class and estimated total batch cost."
        )
        batch_sl  = gr.Slider(0.01, 0.99, value=OPTIMAL_T, step=0.01,
                              label="Confidence threshold")
        batch_btn = gr.Button("Run on all val images", variant="primary")
        batch_md  = gr.Markdown()
        batch_plt = gr.Plot()
        batch_btn.click(run_batch, [batch_sl], [batch_plt, batch_md])

demo.launch(theme=gr.themes.Soft())
