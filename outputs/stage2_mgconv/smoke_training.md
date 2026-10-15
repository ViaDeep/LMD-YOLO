# Stage-2 canonical mg_conv smoke training

Status: **PASS**. One epoch of the full training split and validation of the
serialized `best.pt` completed successfully. This is a functional test, not
an accuracy assessment.

## Architecture

- YAML: `models/yolov8n_leyolo_nano_mgconv_stage2.yaml`
- Backbone: unchanged LeYOLO-Nano stage 1
- Neck: `MGConv` at layers 21 and 24 only
- Main path: inherited 3x3 stride-2 Conv2d with groups=2, BatchNorm2d,
  GELU
- Skip path: ungrouped 1x1 stride-2 Conv2d, added after GELU
- Head: conventional Detect, four classes D00/D10/D20/D40
- Parameters: 2,266,204; GFLOPs @640: 6.165709
- P3/P4/P5: 64x80x80, 128x40x40, 256x20x20; strides 8/16/32

Groups=2 and the projection details are **reproduction decisions**, not
published facts about the original LMD_YOLO model. See
`docs/architecture_decisions.md` and the preserved experimental comparison in
`outputs/stage2_mgconv_experimental`.

## Run

- Data: `data/data.yaml`
- Epochs: 1
- Images: 3,082 train, 385 validation (759 validation objects)
- Image size: 640; batch: 4; device: CPU; workers: 0
- Train/epoch-validation time from `results.csv`: 1,563.51 seconds
- Initial weights: random (`pretrained=False`)
- Training directory: `runs/detect/leyolo_nano_mgconv_stage2_smoke`
- Independent best-checkpoint validation directory:
  `runs/detect/leyolo_nano_mgconv_stage2_best_val`

| Basic metric | Value |
|---|---:|
| Train box loss | 3.68344 |
| Train class loss | 4.76384 |
| Train DFL loss | 3.62812 |
| Validation box loss | 3.34863 |
| Validation class loss | 4.77684 |
| Validation DFL loss | 3.18293 |
| Precision, explicit best.pt validation | 0.00155885 |
| Recall, explicit best.pt validation | 0.23617149 |
| mAP50, explicit best.pt validation | 0.00357338 |
| mAP50-95, explicit best.pt validation | 0.00083303 |

The detection metrics are intentionally not interpreted as a precision claim:
one epoch from random initialization is insufficient. D40 received zero
precision/recall in this run, which is a metric observation, not a pipeline
error. The result establishes that forward, gradients, checkpoint saving,
checkpoint reloading, and validation all work with canonical MGConv.

## Errors and notices

- No exceptions, corrupted images, NaNs, or dimensional errors occurred.
- Ultralytics reported that `optimizer=auto` selected AdamW and ignored its
  default `lr0`/`momentum` values; this also occurred in stage 1.
- Ultralytics displayed an available-version notice. No package was updated.

## Artifacts

- `runs/detect/leyolo_nano_mgconv_stage2_smoke/weights/best.pt`
- `runs/detect/leyolo_nano_mgconv_stage2_smoke/weights/last.pt`
- `runs/detect/leyolo_nano_mgconv_stage2_smoke/results.csv`
- `runs/detect/leyolo_nano_mgconv_stage2_smoke/results.png`
- `outputs/stage2_mgconv/smoke_training.json`

The baseline YOLOv8n and stage-1 LeYOLO-Nano checkpoints retained their
previous sizes and modification times. Detect_DBB was not implemented.
