"""TF-IDF models shared by the analysis scripts (steps 4 and 7)."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from skillshift.metrics import per_source, report, threshold_for_fpr


def word_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)


def fit_word_lr(train, val, test, transform=None):
    """Train word TF-IDF + logistic regression (the step-4 headline model) and score val/test.

    transform: optional function applied to every text before vectorising (used for ablations).
    Returns (result dict, test scores).
    """
    tf = transform or (lambda t: t)
    vec = word_vec()
    x_tr = vec.fit_transform(train.text.map(tf))
    x_va, x_te = vec.transform(val.text.map(tf)), vec.transform(test.text.map(tf))
    model = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0)
    model.fit(x_tr, train.label)
    s_va, s_te = model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]
    t = threshold_for_fpr(val.label, s_va, 0.02)
    result = {
        "train_n": len(train),
        "train_malicious": int(train.label.sum()),
        "default": report(test.label, (s_te >= 0.5).astype(int), s_te),
        "default_per_source": per_source(test, (s_te >= 0.5).astype(int)),
        "val_2pct": report(test.label, (s_te >= t).astype(int), s_te),
    }
    return result, np.asarray(s_te)
