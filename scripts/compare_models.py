"""Step 8: compare TF-IDF and ModernBERT on the same unseen-source test skills, with CIs.

Run (after scripts/mechanism.py and both scripts/run_transformer_suite.sh runs):
    uv run python scripts/compare_models.py
Writes results/compare_models.json.

Every model is scored on the same 1,384 source-disjoint test skills, so a paired, class-stratified
bootstrap gives confidence intervals for each model and for each difference between models.

Metrics:
- benign FPR at the model's default 0.5 threshold (no tuning),
- PR-AUC (threshold-free),
- recall at 2% FPR with the threshold picked on the TEST set: an upper bound, not deployable,
- FPR at 95% and 99% recall (also test-picked), to compare with SkillScreen's matched-recall numbers.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

from skillshift.metrics import bootstrap_ci, recall_at_fpr


def fpr_at_recall(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(fpr[np.argmax(tpr >= target)])


METRICS = {
    "benign_fpr_at_0.5": lambda y, s: float((s[y == 0] >= 0.5).mean()),
    "pr_auc": lambda y, s: float(average_precision_score(y, s)),
    "recall_at_2pct_fpr_oracle": lambda y, s: recall_at_fpr(y, s, 0.02),
    "fpr_at_95_recall": lambda y, s: fpr_at_recall(y, s, 0.95),
    "fpr_at_99_recall": lambda y, s: fpr_at_recall(y, s, 0.99),
}


def main() -> None:
    # TF-IDF columns come from mechanism.py; every transformer/LoRA run drops a
    # <run>_scores.csv into results/transformer/ and is auto-discovered here.
    base = pd.read_csv("results/classical_test_scores.csv")
    names = [c for c in base.columns if c.startswith("word_lr")]
    for path in sorted(Path("results/transformer").glob("*_scores.csv")):
        name = path.stem.replace("_source_disjoint", "").removesuffix("_scores").removesuffix("_s0")
        t = pd.read_csv(path)
        t = t[t.split == "test"][["benchmark_id", "score"]].rename(columns={"score": name})
        base = base.merge(t, on="benchmark_id", how="left")
        names.append(name)
    available = [n for n in names if n in base and base[n].notna().all()]
    y = base.label.to_numpy()
    scores = {n: base[n].to_numpy() for n in available}
    out = {"n_test": len(base), "n_benign": int((y == 0).sum()), "models": available}
    for mname, fn in METRICS.items():
        out[mname] = bootstrap_ci(y, scores, fn)
        print(mname, {k: v for k, v in out[mname].items() if "-" not in k})
    with open("results/compare_models.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
