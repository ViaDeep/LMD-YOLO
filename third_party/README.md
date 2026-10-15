# Vendored Ultralytics source

This directory contains a project-local copy of the installed Ultralytics 8.4.153
Python package. It exists so custom LeYOLO modules can be registered without editing
`.venv/Lib/site-packages`.

Local changes for stage 1 are limited to:

- `ultralytics/nn/modules/leyolo.py`: LeYOLO building blocks.
- `ultralytics/nn/modules/__init__.py`: module exports.
- `ultralytics/nn/tasks.py`: YAML parser registration and expansion-channel scaling.

Stage-2 experimental (not canonical) additions:

- `ultralytics/nn/modules/mg_conv_experimental.py`: two group-count hypotheses
  sharing the same projected-residual topology.
- The module export and YAML parser registration also include
  `ExperimentalMGConv`; the stage-1 YAML and its run artifacts are untouched.

Canonical stage 2 adds `ultralytics/nn/modules/mg_conv.py`, a fixed-groups=2
wrapper around the validated experimental variant B. The authors did not
publish the number of groups; two groups is this project's explicit choice.

Scripts must insert `C:/TP1/third_party` before importing `ultralytics`. The validation
script checks both the import path and package version to prevent accidental mixing.
