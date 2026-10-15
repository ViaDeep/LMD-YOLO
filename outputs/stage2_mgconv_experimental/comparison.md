# Stage 2: experimental mg_conv interpretations (no selection)

Source: Shuai He et al., *LMD_YOLO: A Lightweight and Efficient Model for
Pavement Defects Detection*, PDF supplied at
`C:/Users/Usuario/Downloads/full_text (11).pdf`, especially Section III-D,
Section III-F, Figures 6, 7, 9, and Tables 4, 5.

No variant is canonical. Neither variant was trained. Detect remains the
conventional Ultralytics Detect module; Detect_DBB is not implemented.

## Evidence levels

**Stated in paper text:** `mn_conv` is Conv2d, BatchNorm2d, SiLU, with
`autopad`; the convolution supports groups greater than one. `mg_conv`
replaces SiLU with GELU and adds a residual structure in the neck. The paper
does not give an effective group count for the mg_conv placements.

**Inferred visually from Figure 7:** the main path is autopad, Conv2d,
BatchNorm, GELU. A separate bare Conv2d projects the input directly to an
addition after GELU. The figure supplies no kernel, stride, groups, bias,
padding, or channel specification for that projection.

**Inferred visually from Figure 9:** there are two mg_conv blocks in the
descending neck path, mapped here to stage-1 layers 21 and 24. No other neck
layer is changed.

**Experimental implementation decisions:** the main paths inherit the two
replaced neck convolutions' channels/kernel/stride: 64->64 and 128->128,
kernel 3, stride 2. The residual projection is a bare, ungrouped 1x1 Conv2d
with stride 2 and output channels matching its main path. Both Conv2d
operations keep `bias=True`, consistent with the already-used public
LeYOLO `mn_conv` port, but not specified by LMD_YOLO. Variant A uses one main
group; variant B uses two main groups, the smallest nontrivial group count
valid for both channel pairs. No group count was chosen to fit paper totals.

## Model measurements

All measurements use the same local Ultralytics 8.4.153 profiler at 640x640.

| Model | Main groups | Residual | Parameters | Delta vs stage 1 | GFLOPs | Delta vs stage 1 | Forward | P3/P4/P5 | Strides |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| Stage 1 LeYOLO-Nano | n/a | n/a | 2,337,500 | - | 6.257459 | - | pass | 64x80x80 / 128x40x40 / 256x20x20 | 8/16/32 |
| A, ungrouped | 1 | Conv2d 1x1, s2, g1 | 2,358,364 | +20,864 | 6.283674 | +0.026215 | pass | 64x80x80 / 128x40x40 / 256x20x20 | 8/16/32 |
| B, minimally grouped | 2 | Conv2d 1x1, s2, g1 | 2,266,204 | -71,296 | 6.165709 | -0.091750 | pass | 64x80x80 / 128x40x40 / 256x20x20 | 8/16/32 |

Both candidates build, forward a 1x3x640x640 tensor, keep Detect.nc=4 and
the D00/D10/D20/D40 data configuration, and align the residual shapes:
layer 21 outputs 1x64x40x40 on both paths; layer 24 outputs 1x128x20x20.
The two experimental YAML files are structurally identical to the stage-1
YAML except for neck layers 21 and 24.

## Relation to paper ablation

Table 5 reports 1.39 M / 3.9 G for YOLOv8n-LeYOLO and 1.34 M / 3.8 G when
mg_conv is added: a change of -0.05 M / -0.1 G. Table 4 also gives 1.34 M /
3.8 G for mg_conv, while a prose paragraph says 1.36 M / 3.8 G. The table
figures are used as the principal reference, not optimization targets.

Variant A moves in the opposite direction (+0.020864 M / +0.026215 G).
Variant B moves in the same direction (-0.071296 M / -0.091750 G). These
relative changes are informative but cannot identify the paper's unpublished
implementation, especially because our stage-1 baseline starts at 2.3375 M /
6.257459 G rather than the paper's 1.39 M / 3.9 G.

Variant B is the better motivated *experimental hypothesis*: it exercises the
paper's discussed grouped-convolution capability, uses the smallest valid
nontrivial grouping, preserves the Figure 7 topology, and has the same cost
direction as the ablation. This is not a definitive selection or a claim that
the authors used two groups. User authorization is required before making any
variant canonical or running a smoke training.
