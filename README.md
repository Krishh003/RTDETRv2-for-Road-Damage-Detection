# RT-DETRv2 — Road Damage Detection

Fine-tuning and extension workspace for RT-DETRv2, applied to pothole and road damage detection on the RDD2022 dataset.

Based on:
- "DETRs Beat YOLOs on Real-time Object Detection" (arXiv:2304.08069)
- "RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer" (arXiv:2407.17140)

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `RT-DETR/rtdetrv2_pytorch/` | Upstream PyTorch implementation (primary working directory) |
| `RT-DETR/rtdetrv2_pytorch/delta_additions/` | Custom scripts added on top of the base implementation |
| `RT-DETR/rtdetrv2_pytorch/output/` | Training checkpoints (gitignored) |
| `RT-DETR/rtdetrv2_pytorch/dataset/` | COCO 2017 and RDD2022 datasets (gitignored) |

### Custom Scripts (`delta_additions/`)

| Script | Purpose |
|--------|---------|
| `delta_inference.py` | Enhanced inference for image/video/webcam with XAI (Eigen-CAM) and NMS |
| `delta_tracking.py` | Multi-object tracking using SORT (Kalman filter + Hungarian matching) |
| `benchmark_performance.py` | FPS and throughput benchmarking across configurations |
| `inference_test_epochs.py` | Evaluate checkpoints across training epochs |
| `plot_results.py` | Plot mAP vs epoch from training logs |

---

## Pretrained Checkpoints

| Model | Backbone | Dataset | AP | AP50 | Params |
|-------|----------|---------|-----|------|--------|
| RT-DETRv2-S (pretrained) | R18 | COCO | 48.1 | 65.1 | 20M |

Starting checkpoint: `rtdetrv2_r18vd_120e_coco_rerun_48.1.pth`

### Fine-tuning Runs (RDD2022 / pothole)

| Run | Config | Notes |
|-----|--------|-------|
| `full_finetune` | pothole_detection.yml, r18vd | Full model fine-tune, 1-class pothole |
| `full_finetune_480` | pothole, 480px input | Lower resolution for latency testing |
| `lora_pothole` | pothole + LoRA rank 16 | Memory-efficient, frozen backbone |

---

## Setup

```bash
cd RT-DETR/rtdetrv2_pytorch
pip install -r requirements.txt
```

PyTorch compatibility:

| torch | torchvision |
|-------|-------------|
| 2.4   | 0.19        |
| 2.2   | 0.17        |
| 2.1   | 0.16        |
| 2.0   | 0.15        |

For tracking:
```bash
pip install filterpy scipy
```

Docker:
```bash
docker compose up
```

---

## Usage

All commands run from `RT-DETR/rtdetrv2_pytorch/`.

### Fine-tune on RDD2022

Full fine-tune:
```bash
python tools/train.py \
    -c configs/dataset/pothole_detection.yml \
    -t rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
    -u num_classes=1 remap_mscoco_category=False \
    --use-amp --seed=0
```

LoRA fine-tune (lower memory):
```bash
python tools/train.py \
    -c configs/dataset/pothole_detection.yml \
    -t rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
    -u num_classes=1 remap_mscoco_category=False \
    --lora --lora-rank 16 --use-amp --seed=0
```

### Evaluation

```bash
python tools/train.py \
    -c configs/dataset/pothole_detection.yml \
    -r output/full_finetune/best.pth \
    --test-only
```

Across multiple epoch checkpoints:
```bash
python delta_additions/inference_test_epochs.py
```

### Inference

```bash
python delta_additions/delta_inference.py \
    --config configs/dataset/pothole_detection.yml \
    --weights output/full_finetune/best.pth \
    --input path/to/image_or_video \
    --threshold 0.4
```

### Tracking

```bash
python delta_additions/delta_tracking.py \
    -c configs/dataset/pothole_detection.yml \
    -r output/full_finetune/best.pth \
    --input path/to/video.mp4
```

Controls: `q` to quit, `Space` to pause/resume.

### Export

ONNX:
```bash
python tools/export_onnx.py \
    -c configs/dataset/pothole_detection.yml \
    -r output/full_finetune/best.pth \
    -o model.onnx --check
```

TensorRT (from ONNX):
```bash
python tools/export_trt.py -i model.onnx
```

---

## Dataset Setup

### Road Damage Dataset (RDD2022)

```
dataset/rdd2022/
  train/images/
  val/images/
  annotations/
    instances_train.json
    instances_val.json
```

Class modes:
- 1-class: pothole only (`num_classes: 1`)
- 4-class: D00 (longitudinal crack), D10 (transverse crack), D20 (alligator crack), D40 (pothole)

Convert YOLO annotations to COCO format:
```bash
python dataset/yolo_to_coco.py
```

### COCO 2017

```
dataset/coco2017/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

---

## Config System

YAML configs with `__include__` inheritance. Override on the command line with `-u key=value`:

```bash
python tools/train.py -c configs/dataset/pothole_detection.yml \
    -u epoches=50 num_classes=4
```

Key configs:
- `configs/dataset/pothole_detection.yml` — RDD2022 dataset config
- `configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml` — base model (R18, 120 epochs)
- `configs/runtime.yml` — checkpoint cadence, EMA, AMP
- `configs/rtdetrv2/include/optimizer.yml` — learning rate, scheduler

---

## Checkpoint Diagnostics

```bash
# Inspect a checkpoint
python diagnose_models.py output/full_finetune/best.pth

# Compare two checkpoints
python compare_models.py output/full_finetune/best.pth output/lora_pothole/best.pth
```

---

## Open Tasks

- [ ] Run `inference_test_epochs.py` across all fine-tune runs and compile mAP table
- [ ] Formal LoRA vs full fine-tune AP comparison on RDD2022 val set
- [ ] Export best checkpoint to ONNX and verify end-to-end
- [ ] Evaluate 4-class RDD2022 mode (re-export annotations, retrain)
- [ ] FPS vs AP trade-off analysis for 480px run

---

## Citation

```bibtex
@misc{lv2023detrs,
  title={DETRs Beat YOLOs on Real-time Object Detection},
  author={Wenyu Lv et al.},
  year={2023},
  eprint={2304.08069},
  archivePrefix={arXiv}
}

@misc{lv2024rtdetrv2,
  title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer},
  author={Wenyu Lv et al.},
  year={2024},
  eprint={2407.17140},
  archivePrefix={arXiv}
}
```
