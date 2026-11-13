"""Step 4: can training choices make TF-IDF models less dependent on source style?

Run: uv run python scripts/robust_classical.py
Writes results/robust_classical.json. Evaluated on source_disjoint (unseen test sources).

Variants:
- all:            train on everything (the paper's setting).
- group_balanced: weight samples so every (source, label) group counts equally,
                  so the 3,088 MalSkillBench skills can't dominate.
- mixed_only:     train only on sources that contain BOTH labels (SkillTrustBench,
                  AgentTrap, SkillTrojan). Inside one source, style can't separate
                  the labels, so the model has to use content.
- size_matched_sN: control. A random subset of all training data, the same size and
                  label ratio as mixed_only (seeds 0-2). If it does as well as mixed_only,
                  the gain is just "less data", not "the right data".

Thresholds: "default" uses the model's own 0/0.5 cut. "val_2pct" picks the threshold that
flags 2% of validation benign skills and applies it to test unchanged, which is how you'd
deploy it. Validation comes from training sources, so this tests whether calibration transfers.
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, "src")
from skillshift.data import load_split  # noqa: E402
from skillshift.metrics import per_source, report, threshold_for_fpr  # noqa: E402

MIXED_SOURCES = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}


def group_weights(df):
    groups = df.source_name + "|" + df.label.astype(str)
    counts = groups.map(groups.value_counts())
    w = 1.0 / counts
    return (w / w.mean()).to_numpy()


def word_vec():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)


def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=300_000, sublinear_tf=True)


def main() -> None:
    df = load_split("source_disjoint")
    train_all, val_all, test = (df[df.split == s] for s in ("train", "validation", "test"))
    results = {}

    mixed_train = train_all[train_all.source_name.isin(MIXED_SOURCES)]
    variants = ["all", "group_balanced", "mixed_only"] + [f"size_matched_s{i}" for i in range(3)]
    for variant in variants:
        if variant == "mixed_only":
            train = mixed_train
            val = val_all[val_all.source_name.isin(MIXED_SOURCES)]
        elif variant.startswith("size_matched"):
            seed = int(variant[-1])
            train = pd.concat([
                train_all[train_all.label == y].sample((mixed_train.label == y).sum(), random_state=seed)
                for y in (0, 1)
            ])
            val = val_all
        else:
            train, val = train_all, val_all
        weights = group_weights(train) if variant == "group_balanced" else None

        for feats, make_vec in [("word", word_vec), ("char", char_vec)]:
            vec = make_vec()
            x_tr = vec.fit_transform(train.text)
            x_va, x_te = vec.transform(val.text), vec.transform(test.text)
            models = {
                "lr": LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0),
                "svm": LinearSVC(class_weight="balanced"),
            }
            for mname, model in models.items():
                model.fit(x_tr, train.label, sample_weight=weights)
                score = (model.predict_proba if mname == "lr" else model.decision_function)
                s_va = score(x_va)[:, 1] if mname == "lr" else score(x_va)
                s_te = score(x_te)[:, 1] if mname == "lr" else score(x_te)
                default_cut = 0.5 if mname == "lr" else 0.0
                t = threshold_for_fpr(val.label, s_va, 0.02)
                key = f"{variant}/{feats}_{mname}"
                results[key] = {
                    "train_n": len(train),
                    "default": report(test.label, (s_te >= default_cut).astype(int), s_te),
                    "val_2pct": report(test.label, (s_te >= t).astype(int), s_te),
                    "val_2pct_per_source": per_source(test, (s_te >= t).astype(int)),
                }
                d, v = results[key]["default"], results[key]["val_2pct"]
                print(f"{key:28s} F1 {d['macro_f1']:.3f} FPR {d['benign_fpr']:.3f} "
                      f"PR-AUC {d['pr_auc']:.3f} | val-2% cut: test FPR {v['benign_fpr']:.3f} "
                      f"recall {v['malicious_recall']:.3f}")

    with open("results/robust_classical.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
