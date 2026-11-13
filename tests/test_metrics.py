import numpy as np
import pandas as pd

from skillshift.metrics import per_source, recall_at_fpr, report, threshold_for_fpr


def test_threshold_for_fpr_caps_false_positives():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(1000)]
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(2, 1, 1000)]
    t = threshold_for_fpr(y, s, 0.02)
    assert (s[y == 0] >= t).mean() <= 0.02
    assert (s[y == 0] >= t).mean() > 0.01  # not needlessly strict


def test_recall_at_fpr_perfect_and_useless():
    y = np.array([0, 0, 1, 1])
    assert recall_at_fpr(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.0) == 1.0
    assert recall_at_fpr(y, np.array([0.9, 0.8, 0.1, 0.2]), 0.0) == 0.0


def test_report_counts_benign_false_positives():
    r = report([0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 1, 0], [0.9, 0.1, 0.1, 0.1, 0.8, 0.2])
    assert r["benign_fpr"] == 0.25 and r["malicious_recall"] == 0.5


def test_per_source_splits_by_source():
    df = pd.DataFrame({"source_name": ["A", "A", "B"], "label": [0, 1, 1]})
    out = per_source(df, [1, 1, 0])
    assert out["A"] == {"benign_n": 1, "benign_fpr": 1.0, "malicious_n": 1, "malicious_recall": 1.0}
    assert out["B"] == {"malicious_n": 1, "malicious_recall": 0.0}
