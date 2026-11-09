#!/usr/bin/env bash
# Train the leg/arm detectors (milestone 2) on the datasets from swimsync.dataset.export_yolo.
#   scripts/train_limbs.sh            both versions (4 classes, then 2 classes)
#   scripts/train_limbs.sh 4cls       one version
# Results: experiments/limbs/output/<version>/ (weights/best.pt, results.csv, plots)
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-mps}"   # college GPU: DEVICE=0 scripts/train_limbs.sh
for v in "${@:-4cls 2cls}"; do
  for name in $v; do
    # fliplr=0: mirroring the image swaps her left and right, but not the labels
    .venv/bin/yolo detect train data="data/raw/datasets/m2_${name}/data.yaml" model=yolo11s.pt \
      imgsz=960 epochs=100 patience=30 batch=8 fliplr=0.0 seed=0 device="$DEVICE" \
      project="$PWD/experiments/limbs/output" name="$name" exist_ok=True
  done
done
