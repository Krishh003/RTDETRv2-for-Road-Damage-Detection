"""
generate_training_curves.py
Reads RT-DETRv2 training logs and outputs Figure 5 (training curves)
for the CVIP IEEE paper.

Usage:
    python paper/generate_training_curves.py

Outputs:
    paper/figures/fig5_training_curves.pdf
    paper/figures/fig5_training_curves.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "figures")
LOG_SINGLE  = os.path.join(
    SCRIPT_DIR, "..", "RT-DETR", "rtdetrv2_pytorch",
    "output", "rdd2022_single", "log.txt"
)
LOG_480     = os.path.join(
    SCRIPT_DIR, "..", "RT-DETR", "rtdetrv2_pytorch",
    "output", "road_damage_480", "log.txt"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_log(path):
    """Return list of dicts with epoch, AP50, AP, AR from a log.txt file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # keep only eval rows (they have 'test_coco_eval_bbox')
            if "test_coco_eval_bbox" not in entry:
                continue
            bbox = entry["test_coco_eval_bbox"]
            # COCO eval stats order:
            # [AP, AP50, AP75, APsmall, APmed, APlarge,
            #  AR@1, AR@10, AR@100, ARsmall, ARmed, ARlarge]
            if len(bbox) < 9:
                continue
            records.append({
                "epoch":   entry.get("epoch", len(records)),
                "AP":      bbox[0],   # mAP50-95
                "AP50":    bbox[1],   # mAP50
                "AR100":   bbox[8],   # AR@100 (averaged over IoU 0.50:0.95)
            })
    return records


def main():
    single = parse_log(LOG_SINGLE)
    rdm480 = parse_log(LOG_480)

    if not single:
        raise RuntimeError(f"No eval records found in {LOG_SINGLE}")

    ep_s  = [r["epoch"] for r in single]
    ap50_s = [r["AP50"] * 100 for r in single]
    ap_s   = [r["AP"]   * 100 for r in single]

    ep_480  = [r["epoch"] for r in rdm480]
    ap50_480 = [r["AP50"] * 100 for r in rdm480]

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    fig.subplots_adjust(wspace=0.35)

    GREY  = "#555555"
    BLUE  = "#2166ac"
    RED   = "#d6604d"
    GREEN = "#4dac26"

    # ── Left panel: rdd2022_single mAP50 and mAP50-95 ─────────────────────────
    ax = axes[0]
    ax.plot(ep_s, ap50_s, color=BLUE,  linewidth=1.6, label="mAP50")
    ax.plot(ep_s, ap_s,   color=RED,   linewidth=1.6, label="mAP50-95", linestyle="--")
    ax.axhline(y=62.5, color=GREY, linewidth=1.0, linestyle=":", label="RDD-YOLO mAP50 (62.5%)")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("AP (%)", fontsize=9)
    ax.set_title("(a) rdd2022\_single — AP vs.\\ Epoch", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # ── Right panel: road_damage_480 mAP50 convergence ────────────────────────
    ax2 = axes[1]
    ax2.plot(ep_480, ap50_480, color=GREEN, linewidth=1.6, label="mAP50 (road\_damage\_480)")
    ax2.axhline(y=62.5, color=GREY, linewidth=1.0, linestyle=":", label="RDD-YOLO mAP50 (62.5%)")
    ax2.set_xlabel("Epoch", fontsize=9)
    ax2.set_ylabel("mAP50 (%)", fontsize=9)
    ax2.set_title("(b) road\_damage\_480 — mAP50 vs.\\ Epoch", fontsize=9)
    ax2.legend(fontsize=7.5, loc="lower right")
    ax2.set_xlim(left=0)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax2.tick_params(labelsize=8)
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # ── Save ──────────────────────────────────────────────────────────────────
    for fmt in ("pdf", "png"):
        out = os.path.join(OUTPUT_DIR, f"fig5_training_curves.{fmt}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
