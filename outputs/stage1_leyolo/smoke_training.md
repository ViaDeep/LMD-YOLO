# LeYOLO-Nano stage-1 smoke training

- Status: completed successfully
- Model: `models/yolov8n_leyolo_nano_stage1.yaml`
- Data: `data/data.yaml`
- Epochs: 1
- Image size: 640
- Batch: 4
- Device: CPU
- Workers: 0
- Training images: 3,082
- Validation images: 385 (759 objects)
- Total reported time: 1,630.61 seconds

## Epoch 1

| Metric | Value |
|---|---:|
| train box loss | 3.76387 |
| train class loss | 4.70849 |
| train DFL loss | 3.70521 |
| validation box loss | 3.42417 |
| validation class loss | 4.73899 |
| validation DFL loss | 3.24903 |
| precision | 0.00145 |
| recall | 0.20185 |
| mAP50 | 0.00196 |
| mAP50-95 | 0.00037 |

The low detection metrics are expected for a randomly initialized model after one
epoch and are not used as an accuracy conclusion. The purpose of this run was to
verify end-to-end training, checkpoint serialization, and validation.

## Artifacts

- Run directory: `runs/detect/leyolo_nano_stage1_smoke`
- Best checkpoint: `runs/detect/leyolo_nano_stage1_smoke/weights/best.pt`
- Last checkpoint: `runs/detect/leyolo_nano_stage1_smoke/weights/last.pt`
- Metrics: `runs/detect/leyolo_nano_stage1_smoke/results.csv`

The existing `runs/detect/train` YOLOv8n baseline was not modified. No `mg_conv`
implementation is included in this run.
