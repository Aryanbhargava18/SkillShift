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


def bootstrap_ci(y, scores_by_model, metric, n_boot=2000, seed=0, alpha=0.05):
    """Paired, class-stratified bootstrap of a test-set metric.

    scores_by_model: {name: scores} for models evaluated on the same test skills.
    metric(y, scores) -> float.
    Each resample draws benign and malicious skills separately (class sizes stay fixed)
    and evaluates every model on the same resample, so differences between models are paired.
    Returns {name: [point, low, high]} plus {"a-b": [...]} for every ordered pair a, b.
    """
    y = np.asarray(y)
    scores_by_model = {k: np.asarray(v) for k, v in scores_by_model.items()}
    rng = np.random.default_rng(seed)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    names = list(scores_by_model)
    boots = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(idx0, len(idx0)), rng.choice(idx1, len(idx1))])
        for k in names:
            boots[k][b] = metric(y[idx], scores_by_model[k][idx])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {}
    for k in names:
        point = metric(y, scores_by_model[k])
        out[k] = [round(point, 4), round(float(np.percentile(boots[k], lo)), 4),
                  round(float(np.percentile(boots[k], hi)), 4)]
    for a in names:
        for b_ in names:
            if a != b_:
                d = boots[a] - boots[b_]
                out[f"{a}-{b_}"] = [round(out[a][0] - out[b_][0], 4),
                                    round(float(np.percentile(d, lo)), 4),
                                    round(float(np.percentile(d, hi)), 4)]
    return out


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial confidence interval for k successes out of n (for small groups, e.g. n=21)."""
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(k / n, 4), round(lo, 4), round(hi, 4)]


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
