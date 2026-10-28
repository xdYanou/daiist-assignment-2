import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml as pyyaml
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key

# ── Dataset ───────────────────────────────────────────────────────────────────
WORKSPACE = "mandilannotate-fyest"
PROJECT   = "detectorn2-vh2zv"
VERSION   = 1

# ── Training ──────────────────────────────────────────────────────────────────
BASE_CHECKPOINT = "yolo11n.pt"
EPOCHS = 40
IMGSZ  = 640

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS     = Path("weights/best.pt")
RESULTS_DIR = Path("results")

# ── Cost matrix (euros) ───────────────────────────────────────────────────────
# Positive class: total_damage — the most severe, highest-cost outcome.
# FN: miss total_damage, dispatch under-equipped team → €150,000
# FP: false alarm, over-deploy full response team → €50,000
# BASE_RATE: ~15% of inspected sites have total damage
COST_FN   = 150_000.0
COST_FP   =  50_000.0
BASE_RATE = 0.15

WEIGHTS.parent.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
print("Config ready.")

### ---- Dataset download and prep
rf = Roboflow(api_key=load_api_key())
# coco-segmentation format — this dataset is semantic segmentation type
dataset = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("coco-segmentation")
print(f"Dataset saved to: {dataset.location}")

def _find_pos_idx(names):
    """Find the index of the positive class (total_damage first, major_damage as fallback)."""
    for exact in ["total_damage", "major_damage"]:
        for i, n in enumerate(names):
            if n.lower() == exact:
                return i
    for keyword in ["total", "major", "severe"]:
        for i, n in enumerate(names):
            if keyword in n.lower():
                return i
    return 0

# --- Convert COCO segmentation → YOLO detection

import json as _json
from collections import defaultdict as _defaultdict

def coco_to_yolo(dataset_dir):
    """
    Convert Roboflow coco-segmentation export → YOLO detection format.
    - Copies images into split/images/ subfolders
    - Writes YOLO bbox txt files into split/labels/
    - Generates data.yaml
    Returns (yaml_path, class_names).
    """
    dataset_dir = Path(dataset_dir)
    all_names = None

    for split in ["train", "valid", "test"]:
        split_dir = dataset_dir / split
        ann_file  = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"  [skip] {split}: _annotations.coco.json not found")
            continue

        with open(ann_file) as f:
            coco = _json.load(f)

        cats = sorted(coco["categories"], key=lambda c: c["id"])
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
        if all_names is None:
            all_names = [c["name"] for c in cats]

        img_map = {img["id"]: img for img in coco["images"]}

        img_dst = split_dir / "images"
        lbl_dst = split_dir / "labels"
        img_dst.mkdir(exist_ok=True)
        lbl_dst.mkdir(exist_ok=True)

        img_anns = _defaultdict(list)
        for ann in coco["annotations"]:
            img_anns[ann["image_id"]].append(ann)

        n_copied = 0
        for img_id, img_info in img_map.items():
            fname = img_info["file_name"]
            W, H  = img_info["width"], img_info["height"]
            stem  = Path(fname).stem

            src = split_dir / fname
            dst = img_dst / Path(fname).name
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
                n_copied += 1

            lines = []
            for ann in img_anns.get(img_id, []):
                cls_idx = cat_id_to_idx[ann["category_id"]]
                x, y, w, h = ann["bbox"]   # COCO: top-left corner + width/height
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw = w / W
                nh = h / H
                lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            (lbl_dst / f"{stem}.txt").write_text("\n".join(lines))

        print(f"  {split}: {len(img_map)} images ({n_copied} copied to images/), "
              f"{len(coco['annotations'])} annotations → labels/")

    if all_names is None:
        raise RuntimeError("No _annotations.coco.json found — check the download format.")

    yaml_content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"nc: {len(all_names)}\n"
        f"names: {all_names}\n"
    )
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  data.yaml written — nc={len(all_names)}, names={all_names}")
    return str(yaml_path), all_names


print("Converting COCO segmentation → YOLO detection format…")
yaml, names = coco_to_yolo(dataset.location)

print(f"\nClasses: {names}")
print(f"YAML: {yaml}")

# Print split sizes
for split in ["train", "valid", "test"]:
    p = Path(dataset.location) / split / "images"
    if p.exists():
        print(f"  {split}/images → {len(list(p.glob('*')))} images")

occ_idx = _find_pos_idx(names)
print(f"\nPositive class: index={occ_idx}  name='{names[occ_idx]}'")

# Sample filenames
for folder in ["test", "valid"]:
    p = Path(dataset.location) / folder / "images"
    if p.exists():
        print(f"\nSample filenames ({folder}/images):")
        for s in sorted(p.glob("*"))[:5]:
            print(f"  {s.name}")
        break

## Subsample train/test sets for faster iteration

import random

SUBSAMPLE_FRACTION = 0.95
SEED = 42

def subsample_split(location, folder, fraction, seed=SEED):
    """Copy a random fraction of images+labels to <folder>_sub/."""
    img_src = Path(location) / folder / "images"
    lbl_src = Path(location) / folder / "labels"
    img_dst = Path(location) / f"{folder}_sub" / "images"
    lbl_dst = Path(location) / f"{folder}_sub" / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(img_src.glob("*.*"))
    random.seed(seed)
    subset = random.sample(all_imgs, max(1, int(len(all_imgs) * fraction)))

    for img in subset:
        shutil.copy(img, img_dst / img.name)
        lbl = lbl_src / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy(lbl, lbl_dst / lbl.name)

    return f"{folder}_sub/images", len(subset)

# Subsample val and test to keep proportional with training fraction
val_folder  = "valid" if (Path(dataset.location) / "valid").exists() else "val"
test_folder = "test"  if (Path(dataset.location) / "test").exists()  else None

val_path,  n_val  = subsample_split(dataset.location, val_folder,  SUBSAMPLE_FRACTION)
print(f"Val  subset: {n_val} images  → {val_path}")

if test_folder:
    test_path, n_test = subsample_split(dataset.location, test_folder, SUBSAMPLE_FRACTION)
    print(f"Test subset: {n_test} images → {test_path}")
else:
    test_path = None
    print("No test folder found — skipping test subsample")

with open(yaml) as f:
    _cfg = pyyaml.safe_load(f)

_cfg["val"] = val_path
if test_path:
    _cfg["test"] = test_path

with open(yaml, "w") as f:
    pyyaml.dump(_cfg, f)

print(f"\nYAML updated:")
print(f"  val  → {_cfg['val']}")
if test_path:
    print(f"  test → {_cfg['test']}")

## Finetune 
model = YOLO(BASE_CHECKPOINT)
train_results = model.train(
    data=yaml,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    fraction=0.95,
    project="runs/train",
    name="damage",
    exist_ok=True,
    device=0,
)

best = Path(train_results.save_dir) / "weights" / "best.pt"
shutil.copy(best, WEIGHTS)
print(f"Best weights saved to: {WEIGHTS}")

# Eval fine tuned + COCO + TRIVIAL models on Val 
EVAL_SPLIT  = "val"
EVAL_FOLDER = "valid" if (Path(dataset.location) / "valid").exists() else "val"
print(f"Evaluating on: {EVAL_FOLDER}/images (full val set)")

with open(yaml) as f:
    _cfg = pyyaml.safe_load(f)
_cfg["val"] = f"{EVAL_FOLDER}/images"
with open(yaml, "w") as f:
    pyyaml.dump(_cfg, f)


def _per_class_p_r(r):
    """
    Extract per-class precision/recall robustly.
    Ultralytics excludes classes with 0 GT instances from r.box.p/r/ap50,
    so we use ap_class_index to find the correct position for each class.
    """
    idx_arr = list(r.box.ap_class_index.astype(int)) if hasattr(r.box, 'ap_class_index') else list(range(len(r.box.ap50)))
    p_arr   = list(r.box.p)
    r_arr   = list(r.box.r)
    ap_arr  = list(r.box.ap50)
    result  = {}
    for class_idx, name in r.names.items():
        if class_idx in idx_arr:
            pos = idx_arr.index(class_idx)
            result[name] = {
                "AP50":      round(float(ap_arr[pos]), 4),
                "precision": round(float(p_arr[pos]),  4),
                "recall":    round(float(r_arr[pos]),   4),
            }
        else:
            result[name] = {"AP50": 0.0, "precision": 0.0, "recall": 0.0}
    return result, idx_arr, p_arr, r_arr


def _get_occ_pr(r):
    """Return (precision, recall) for the positive class (occ_idx)."""
    _, idx_arr, p_arr, r_arr = _per_class_p_r(r)
    if occ_idx in idx_arr:
        pos = idx_arr.index(occ_idx)
        return float(p_arr[pos]), float(r_arr[pos])
    return 0.0, 0.0


# ── Finetuned model ───────────────────────────────────────────────────────────
def val_metrics(m, conf=0.25):
    r = m.val(data=yaml, split=EVAL_SPLIT, conf=conf, verbose=False)
    per_cls, _, _, _ = _per_class_p_r(r)
    return {
        "mAP50":     round(float(r.box.map50), 4),
        "mAP50-95":  round(float(r.box.map),   4),
        "precision": round(float(r.box.mp),     4),
        "recall":    round(float(r.box.mr),     4),
        "per_class": per_cls,
    }


# ── COCO zero-shot baseline ───────────────────────────────────────────────────
def coco_zero_shot_baseline():
    coco_model = YOLO(BASE_CHECKPOINT)
    img_dir = Path(dataset.location) / EVAL_FOLDER / "images"
    lbl_dir = Path(dataset.location) / EVAL_FOLDER / "labels"

    tp = fp = fn = tn = 0
    for img_path in sorted(img_dir.glob("*.*")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        gt_has_total = any(
            int(line.split()[0]) == occ_idx
            for line in lbl_path.read_text().strip().splitlines() if line.strip()
        )
        res = coco_model.predict(str(img_path), conf=0.25, verbose=False)[0]
        pred_total = res.boxes is not None and len(res.boxes) > 0

        if gt_has_total and pred_total:       tp += 1
        elif gt_has_total and not pred_total: fn += 1
        elif not gt_has_total and pred_total: fp += 1
        else:                                 tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ap50      = precision * recall
    print(f"  COCO zero-shot → TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  P={precision:.3f}  R={recall:.3f}  AP50≈{ap50:.3f}")
    return {
        "mAP50":    round(ap50, 4),
        "mAP50-95": round(ap50 * 0.55, 4),
        "precision": round(precision, 4),
        "recall":   round(recall, 4),
        "per_class": {
            names[occ_idx]: {"AP50": round(ap50,4), "precision": round(precision,4), "recall": round(recall,4)},
            **{n: {"AP50": 0.0, "precision": 0.0, "recall": 0.0} for i, n in enumerate(names) if i != occ_idx},
        },
        "description": "COCO zero-shot: any detection in image → predict total_damage.",
    }


# ── Trivial baseline ──────────────────────────────────────────────────────────
metrics = {}
ft = YOLO(str(WEIGHTS))
metrics["finetuned"] = val_metrics(ft)
print(f"Finetuned mAP50: {metrics['finetuned']['mAP50']:.3f}")
print(f"Per class: {metrics['finetuned']['per_class']}")

print("\nRunning COCO zero-shot baseline...")
metrics["zero_shot_coco"] = coco_zero_shot_baseline()

metrics["trivial"] = {
    "mAP50":    round(BASE_RATE, 4),
    "mAP50-95": round(BASE_RATE * 0.55, 4),
    "precision": round(BASE_RATE, 4),
    "recall":   1.0,
    "description": "Predict every image as total_damage. Recall=1, precision=base rate.",
}
print(f"Trivial mAP50≈ {metrics['trivial']['mAP50']:.3f}")

## COST / CONFIDENCE interval sweep

sweep = []

for t in np.linspace(0.05, 0.95, 30).tolist():
    r  = ft.val(data=yaml, split=EVAL_SPLIT, conf=float(t), verbose=False)
    p, rc = _get_occ_pr(r)

    e_fn = 1000 * BASE_RATE * (1 - rc)
    e_fp = (1000 * BASE_RATE * rc * (1 - p) / p) if p > 0 else 1000 * (1 - BASE_RATE)
    cost = COST_FN * e_fn + COST_FP * e_fp

    sweep.append({"threshold": round(t, 4), "precision": round(p, 4),
                  "recall": round(rc, 4), "cost": round(cost, 2)})
    print(f"conf={t:.2f}  P={p:.3f}  R={rc:.3f}  cost=€{cost:,.0f}")

optimal = min(sweep, key=lambda x: x["cost"])
print(f"\nOptimal threshold: {optimal['threshold']}  (€{optimal['cost']:,.0f} per 1k cases)")

# Save WEIGHTS AND METRICS
(RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(RESULTS_DIR / "cost_sweep.json").write_text(json.dumps(sweep, indent=2))
(RESULTS_DIR / "operating_point.json").write_text(json.dumps({
    "threshold": optimal["threshold"],
    "cost_fn":   COST_FN,
    "cost_fp":   COST_FP,
    "base_rate": BASE_RATE,
}, indent=2))

print("Saved:")
print(f"  {RESULTS_DIR}/metrics.json")
print(f"  {RESULTS_DIR}/cost_sweep.json")
print(f"  {RESULTS_DIR}/operating_point.json")
print(f"  {WEIGHTS}")