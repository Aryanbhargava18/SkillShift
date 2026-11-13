"""Step 3: is "which benchmark did this come from" learnable from the text?

Run: uv run python scripts/shortcut_probe.py
Writes results/shortcut_probe.json.

If a plain TF-IDF model can name a skill's source benchmark, and most sources have only
one label, then a malicious/benign classifier can score well by recognising sources.
"""

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

from skillshift.data import load_split


def main() -> None:
    out = {}

    # 1. Predict the source benchmark from text (random split, 13-way minus tiny sources).
    df = load_split("random")
    keep = df.source_name.map(df.source_name.value_counts()) >= 20
    df = df[keep]
    train, test = df[df.split == "train"], df[df.split == "test"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(vec.fit_transform(train.text), train.source_name)
    pred = clf.predict(vec.transform(test.text))
    n_src = df.source_name.nunique()
    out["source_id"] = {
        "n_sources": n_src,
        "accuracy": round(float((pred == test.source_name).mean()), 3),
        "balanced_accuracy": round(float(balanced_accuracy_score(test.source_name, pred)), 3),
        "chance_balanced_accuracy": round(1 / n_src, 3),
    }
    print("source identification:", out["source_id"])

    # 2. What does the paper's malicious/benign SVM rely on under source-disjoint?
    df = load_split("source_disjoint")
    train = df[df.split == "train"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    svm = LinearSVC(class_weight="balanced").fit(vec.fit_transform(train.text), train.label)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(svm.coef_[0])
    out["svm_top_malicious_features"] = names[order[-30:]][::-1].tolist()
    out["svm_top_benign_features"] = names[order[:30]].tolist()
    print("top malicious features:", out["svm_top_malicious_features"])
    print("top benign features:", out["svm_top_benign_features"])

    with open("results/shortcut_probe.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
