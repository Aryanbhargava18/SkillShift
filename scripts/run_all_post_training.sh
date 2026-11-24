#!/bin/sh
# Waits until the all-data ModernBERT run finishes and the GPU is free, then runs the whole
# downstream SkillShift pipeline. Safe to launch while training is still going: it blocks until
# training exits so it never competes for memory (the 8 GB MPS OOM lesson).
cd /Users/aryan./Documents/lfxx/skillshift || exit 1
RUN=modernbert_source_disjoint_mixed_only_s0
ALL=results/transformer/modernbert_source_disjoint_all_s0.json
until [ -f "$ALL" ] && ! pgrep -f "train_transformer.py" >/dev/null; do sleep 60; done
sleep 30                         # let MPS memory settle
exec >results/post_training.log 2>&1
set -e
echo "=== mechanism (TF-IDF ablations) $(date)";   uv run --no-sync python scripts/mechanism.py
echo "=== compare_models (bootstrap CIs) $(date)"; uv run --no-sync python scripts/compare_models.py
echo "=== adv_evasion (attack) $(date)";           uv run --no-sync python scripts/adv_evasion.py
echo "=== export_onnx $(date)";                    uv run --no-sync python scripts/export_onnx.py --run "$RUN"
echo "=== eval_chunked $(date)";                   uv run --no-sync python scripts/eval_chunked.py --run "$RUN"
echo "=== make_paper_numbers $(date)";             uv run --no-sync python scripts/make_paper_numbers.py
echo "=== make_tables $(date)";                    uv run --no-sync python scripts/make_tables.py > results/tables.md
echo "=== DONE $(date)"
