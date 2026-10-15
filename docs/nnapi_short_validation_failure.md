# NNAPI short validation — Samsung Galaxy S23

## Verdict

`ABORTED` during NNAPI session creation. No inference, warmup or sustained
benchmark was executed. The NNAPI benchmark gate was not created.

## Frozen inputs

- Device: Samsung Galaxy S23 SM-S911B.
- ONNX Runtime Android: 1.24.3.
- Precision: FP32.
- Input: `[1,3,640,640]`.
- ONNX SHA-256:
  `a68e5c09d2d9c76f0259d9705727f14917f67f2a6dafcedbffbe86d8deed51b8`.
- Canonical preprocessing: unchanged.

Before session creation, the canonical JPEG pipeline passed bit-exactly against
the PC tensor: 1,228,800 elements, `rtol=0`, `atol=0`, zero failures.

## NNAPI capability messages

ONNX Runtime emitted two capability summaries while optimizing nested graph
views:

- 3 candidate NNAPI partitions; 247 of 249 nodes reported supported.
- 3 candidate NNAPI partitions; 255 of 257 nodes reported supported.

These are capability claims, not evidence of executed nodes. Session creation
failed afterwards, so no NNAPI kernel executed and no completed profiler trace
exists.

## Fatal error

`op_builder_helpers.cc:144 AddNnapiSplit count [0] does not evenly divide dimension 3 [128]`

The failure occurred inside `OrtEnvironment.createSession`; consequently:

- session-load time is unavailable;
- output shape and golden output could not be evaluated under NNAPI;
- there are no measured fallbacks;
- no 10+100 benchmark was started.

## Read-only ONNX inspection

The opset-18 graph contains four equal `Split` operations:

| Node | ONNX axis | Input shape | Split input |
|---|---:|---|---|
| `/model.17/Split` | 1 | `[1,128,40,40]` | `[64,64]` |
| `/model.20/Split` | 1 | `[1,64,80,80]` | `[32,32]` |
| `/model.23/Split` | 1 | `[1,128,40,40]` | `[64,64]` |
| `/model.26/Split` | 1 | `[1,256,20,20]` | `[128,128]` |

The error's dimension 3 of size 128 is consistent with NNAPI's NHWC mapping of
one of the channel-axis `Split` nodes whose NCHW channel dimension is 128. The
reported `count [0]` is inconsistent with the ONNX initializer `[64,64]`; this
points to NNAPI EP construction/translation of `Split`, rather than an invalid
division in the frozen ONNX. This is an inference from the error and graph, not
a completed NNAPI profile.

The same `AddNnapiSplit count [0]` failure has been publicly reported for other
Ultralytics FP32 ONNX models using ONNX Runtime NNAPI, with different channel
dimensions. Therefore it is not specific evidence of LeYOLO, mg_conv or DBB.

## Constraint-respecting conclusion

Under ONNX Runtime Android 1.24.3 and the frozen opset-18 ONNX, NNAPI is not a
viable benchmark route because the session cannot be constructed. Retrying by
rewriting `Split`, changing opset/model, changing runtime, relaxing precision,
or altering provider flags would introduce another variable and was not done.

