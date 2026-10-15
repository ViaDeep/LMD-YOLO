# Stage 1 - LeYOLO structural comparison

The paper values (1.39 M parameters and 3.9 GFLOPs at 640) are references only.
No layer or channel was tuned to match them.

| Candidate | Parameters (M) | GFLOPs | Strides | Backbone P3/P4/P5 |
|---|---:|---:|---|---|
| nano | 2.337500 | 6.257459 | [8, 16, 32] | 32x80x80 / 64x40x40 / 96x20x20 |
| small | 2.840100 | 7.390566 | [8, 16, 32] | 48x80x80 / 88x40x40 / 128x20x20 |

Most consistent by the declared rule: **nano**.

This selection does not prove which unpublished configuration the LMD_YOLO authors used.
