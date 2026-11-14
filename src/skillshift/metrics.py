"""Metrics we report for every model. Label 1 = malicious (the positive class)."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_curve


def recall_at_fpr(y, scores, max_fpr):
    """Best malicious recall while wrongly flagging at most max_fpr of benign skills.

    Note: the threshold is picked on the same data we measure on, so this is a
    ceiling for the model, not a deployable number. Later steps pick it on validation.
    """
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def report(y, pred, scores):
    y, pred = np.asarray(y), np.asarray(pred)
    benign = y == 0
    return {
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 3),
        "benign_fpr": round(float(pred[benign].mean()), 3),  # share of benign wrongly flagged
        "malicious_recall": round(float(pred[~benign].mean()), 3),
        "pr_auc": round(float(average_precision_score(y, scores)), 3),
        "recall_at_1pct_fpr": round(recall_at_fpr(y, scores, 0.01), 3),
        "recall_at_2pct_fpr": round(recall_at_fpr(y, scores, 0.02), 3),
    }


def threshold_for_fpr(y, scores, max_fpr):
    """Lowest threshold whose benign false-positive rate on (y, scores) is <= max_fpr.

    Pick it on validation, then apply it unchanged to test: that is the deployable setting.
    """
    benign = np.sort(np.asarray(scores)[np.asarray(y) == 0])[::-1]
    k = int(np.floor(max_fpr * len(benign)))  # benign samples we may flag
    return float(benign[k]) + 1e-9 if k < len(benign) else float(benign[-1])


def per_source(df, pred):
    """Per test source: benign skills wrongly flagged, malicious skills caught."""
    out = {}
    d = df.assign(pred=np.asarray(pred))
    for src, g in d.groupby("source_name"):
        row = {}
        if (g.label == 0).any():
            row["benign_n"] = int((g.label == 0).sum())
            row["benign_fpr"] = round(float(g.pred[g.label == 0].mean()), 3)
        if (g.label == 1).any():
            row["malicious_n"] = int((g.label == 1).sum())
            row["malicious_recall"] = round(float(g.pred[g.label == 1].mean()), 3)
        out[src] = row
    return out
