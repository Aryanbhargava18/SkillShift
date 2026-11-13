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
