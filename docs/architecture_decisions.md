# LMD_YOLO reproduction: architecture decisions

## Stage 1 scope

This stage replaces only the YOLOv8n backbone with LeYOLO. The effective YOLOv8n
neck and Detect head remain unchanged. No `mg_conv`, `Detect_DBB`, long training, or
parameter tuning to match a reported total has been introduced.

## Sources and ambiguity

The LMD_YOLO paper specifies a LeYOLO backbone composed of two `mn_conv` layers,
twelve inverted bottlenecks, and SPPF, while retaining P3/P4/P5 detection scales. It
does not publish the exact variant, complete channel table, YAML, source code, or
weights. Its Table 3 reports 1.39 M parameters and 3.9 GFLOPs for the LeYOLO ablation;
those values are treated only as comparison references.

The public LeYOLO repository provides Nano and Small configurations. Their layer
sequence and channels were transferred explicitly into separate YAML files. The
official implementation's actual semantics were retained where prose and code can
be read differently: the expansion pointwise convolution is omitted only when the
configuration explicitly disables it, and the attention argument is accepted but
unused by that block implementation.

Ultralytics 8.4.153 resolves custom YAML modules in its task parser. A local source
copy is therefore used and registered instead of changing the installed package in
`.venv`.

## Candidate construction

- `yolov8n_leyolo_nano_stage1.yaml`: official LeYOLO-Nano backbone channels and
  expansion channels, followed by the fixed YOLOv8n neck/head.
- `yolov8n_leyolo_small_stage1.yaml`: official LeYOLO-Small width (1.33) materialized
  in the backbone only, followed by the same fixed YOLOv8n neck/head.

The neck is deliberately held fixed so the comparison isolates the backbone choice.
The two YAML files use four detection classes.

## Structural result

At input 640 x 640, using the same Ultralytics profiler for all models:

| Model | Parameters | GFLOPs | Backbone P3/P4/P5 | Detect strides |
|---|---:|---:|---|---|
| YOLOv8n, 4 classes (control) | 3,011,628 | 8.194918 | native YOLOv8n | 8/16/32 |
| LeYOLO-Nano candidate | 2,337,500 | 6.257459 | 32/64/96 channels at 80/40/20 | 8/16/32 |
| LeYOLO-Small candidate | 2,840,100 | 7.390566 | 48/88/128 channels at 80/40/20 | 8/16/32 |

Both candidates build, execute a 640 x 640 forward pass, expose P3/P4/P5 correctly,
and configure Detect for four classes. Nano also completed an unsaved CPU prediction
on one 512 x 512 test image. The untrained model returned no boxes, which is expected
and is not an accuracy test.

## Decision and limitation

LeYOLO-Nano is selected for the next stage because it is the closer of the two
untuned candidates to both 1.39 M and 3.9 GFLOPs. This does not prove that it is the
authors' unpublished configuration. Its remaining gap (0.9475 M parameters and
2.357459 GFLOPs) shows that the reported ablation cannot be reproduced by swapping
one of these public backbones into an otherwise unchanged YOLOv8n neck/head alone.
Possible explanations include an unreported neck/head width choice, a different
LeYOLO revision, or inconsistent reporting. Those possibilities are recorded rather
than silently reverse-engineered to force the totals.

Work stops here before `mg_conv`, as requested.

## Stage 2: canonical mg_conv decision (2026-09-20)

The instruction to stop above describes the completed stage-1 milestone. The
user subsequently authorized stage 2 and selected experimental variant B.
This is the **canonical choice for this reproduction only**. It is not
evidence that the unpublished original LMD_YOLO used two groups.

The paper's text states that `mn_conv` is Conv2d-BatchNorm-SiLU with autopad
and support for grouped convolution. It states that `mg_conv` uses GELU and a
residual path. Figure 7 depicts a main Conv2d-BatchNorm-GELU path and a bare
Conv2d skip path added after GELU. Figure 9 visually supports two mg_conv
locations in the descending neck path. Mapping those symbols to layer indices
21 and 24 is an inference relative to our stage-1 YAML.

Our implementation makes the following explicit reproduction decisions:

- main-path kernel 3, stride 2, and channel counts 64->64 / 128->128 are
  inherited from the replaced stage-1 neck convolutions;
- main-path `groups=2` is the least nontrivial valid grouping that was tested;
- the residual branch is an ungrouped Conv2d with kernel 1, stride 2, and
  matching output channels, allowing shape-compatible addition;
- both Conv2d operations use bias, following the existing LeYOLO `mn_conv`
  port rather than a published LMD_YOLO parameter specification;
- the rest of the LeYOLO-Nano backbone, YOLOv8n neck, and conventional Detect
  head remain unchanged. Detect_DBB is not present.

For input 1x3x640x640, the canonical model gives 2,266,204 parameters and
6.165709 GFLOPs, 71,296 parameters and 0.091750 GFLOPs below our stage-1
model. It produces P3/P4/P5 at 80/40/20 spatial resolution with strides
8/16/32 and four classes. The same measurements and state-dict tensor shapes
were verified against experimental variant B before training.

The paper's Tables 4 and 5 report 1.34 M and 3.8 G for the mg_conv ablation
versus 1.39 M and 3.9 G without it; a paragraph instead says 1.36 M. These
are comparison references only. Our absolute values are not forced to match
an unpublished configuration.

The one-epoch CPU smoke training and an independent validation of the saved
`best.pt` both completed successfully. Their details are recorded in
`outputs/stage2_mgconv/smoke_training.md`; these one-epoch detection metrics
are not used to judge model accuracy.
