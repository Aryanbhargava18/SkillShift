#!/bin/sh
# Runs the three fine-tuning runs reported in the README, one after another (~3.5 h on an M3).
# caffeinate keeps a Mac awake; remove it on Linux.
set -e
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONUNBUFFERED=1
for run in "source_disjoint mixed_only" "source_disjoint all" "random all"; do
  set -- $run
  caffeinate -i uv run python scripts/train_transformer.py --protocol "$1" --variant "$2" \
    2>&1 | grep --line-buffered -v -i -E "warn|Writing model shards" > "results/transformer/logs/$1_$2.log"
done
