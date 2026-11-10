#!/usr/bin/env bash
# Train the leg/arm models (milestone 2) on the datasets from swimsync.dataset.export_yolo
# (boxes: 4cls, 2cls) and swimsync.dataset.limb_masks (outlines: seg).
#   scripts/train_limbs.sh            both box versions (4 classes, then 2 classes)
#   scripts/train_limbs.sh seg        leg/arm outlines (YOLO11s-seg)
# Results: experiments/limbs/output/<version>/ (weights/best.pt, results.csv, plots)
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-mps}"   # college GPU: DEVICE=0 scripts/train_limbs.sh
EPOCHS="${EPOCHS:-100}"   # longer: EPOCHS=200 scripts/train_limbs.sh seg
NAME_SUFFIX="${NAME_SUFFIX:-}"  # e.g. NAME_SUFFIX=_200ep keeps earlier results
for name in ${@:-4cls 2cls}; do
  if [ "$name" = seg ]; then task=segment; model=yolo11s-seg.pt; else task=detect; model=yolo11s.pt; fi
  # fliplr=0: mirroring the image swaps her left and right, but not the labels
  .venv/bin/yolo "$task" train data="data/raw/datasets/m2_${name}/data.yaml" model="$model" \
    imgsz=960 epochs="$EPOCHS" patience=30 batch=8 fliplr=0.0 seed=0 device="$DEVICE" \
    project="$PWD/experiments/limbs/output" name="${name}${NAME_SUFFIX}" exist_ok=True
done
