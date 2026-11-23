#!/usr/bin/env bash
# Train the leg/arm models (milestone 2) on the datasets from swimsync.dataset.export_yolo
# (boxes: 4cls, 2cls) and swimsync.dataset.limb_masks (outlines: seg).
#   scripts/train_limbs.sh            both box versions (4 classes, then 2 classes)
#   scripts/train_limbs.sh seg        leg/arm outlines (YOLO11s-seg)
#   SIZE=m IMGSZ=640 BATCH=8 scripts/train_limbs.sh pose   pose model on the swimmer crops (export_pose)
#   IMGSZ=384 BATCH=32 scripts/train_limbs.sh facing       legs front/back (facing export)
#   IMGSZ=384 BATCH=32 scripts/train_limbs.sh view nearer  which way the legs face, nearer leg (export-view)
# Results: experiments/limbs/output/<version>/ (weights/best.pt, results.csv, plots)
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${DEVICE:-mps}"   # college GPU: DEVICE=0 scripts/train_limbs.sh
EPOCHS="${EPOCHS:-100}"   # longer: EPOCHS=200 scripts/train_limbs.sh seg
NAME_SUFFIX="${NAME_SUFFIX:-}"  # e.g. NAME_SUFFIX=_200ep keeps earlier results
SIZE="${SIZE:-s}"         # model size: SIZE=m for YOLO11m (slower, more memory)
BATCH="${BATCH:-8}"       # images per step; YOLO11m-seg at 960 px needs BATCH=4 on a 24 GB Mac
IMGSZ="${IMGSZ:-960}"     # image size the model sees
for name in ${@:-4cls 2cls}; do
  # fliplr=0: mirroring the image swaps her left and right, but not the box labels.
  # Pose is the exception: flip_idx in its data.yaml swaps the left/right points too.
  # Front/back doesn't change in a mirror, so facing can flip both ways.
  # view/nearer: mirrored copies with swapped answers are already in the dataset, so flips stay off.
  # nearleg (is this leg the nearer one?) only learns with augmentation off (results.md, 2026-11-23).
  flip=0.0; flipud=0.0; extra=""; data="data/raw/datasets/m2_${name}/data.yaml"
  case "$name" in
    seg) task=segment; model="yolo11${SIZE}-seg.pt" ;;
    pose*) task=pose; model="yolo11${SIZE}-pose.pt"; flip=0.5 ;;
    facing*) task=classify; model="yolo11${SIZE}-cls.pt"; flip=0.5; flipud=0.5; data="data/raw/datasets/m2_facing" ;;
    nearleg*) task=classify; model="yolo11${SIZE}-cls.pt"; data="data/raw/datasets/m2_nearleg"
      extra="scale=0 erasing=0 auto_augment=None hsv_h=0 hsv_s=0 hsv_v=0" ;;
    view*|nearer*) task=classify; model="yolo11${SIZE}-cls.pt"; data="data/raw/datasets/m2_${name%%_*}" ;;
    *) task=detect; model="yolo11${SIZE}.pt" ;;
  esac
  .venv/bin/yolo "$task" train data="$data" model="$model" \
    imgsz="$IMGSZ" epochs="$EPOCHS" patience=30 batch="$BATCH" fliplr="$flip" flipud="$flipud" seed=0 device="$DEVICE" \
    project="$PWD/experiments/limbs/output" name="${name}${NAME_SUFFIX}" exist_ok=True $extra
done
