"""Step 7: what actually drives the mixed-only gain?

Run: uv run python scripts/mechanism.py
Writes results/mechanism.json. All models are word TF-IDF + logistic regression (the step-4
headline model), trained on source_disjoint train and evaluated on the unseen test sources.

Step 4 showed that training only on mixed-label sources cuts benign false positives on unseen
sources from 65% to 3.7%. But mixed-only training is ~97% SkillTrustBench, and SkillTrustBench
builds most malicious skills by injecting a payload into a real benign skill. So "mixed-only"
could work because of label mixing inside a source, or because the data is made of minimal pairs
(benign skill + the same skill with a payload). This script separates the two, and checks the
other explanations reviewers would ask about.

1. pairing:      how many mixed-source malicious skills have a benign twin (same frontmatter name),
                 and how much of the twin's text they keep.
2. break_pairs:  keep only ONE side (benign or malicious, coin flip) of every twin group, so no pair
                 survives. Control: random subsets of mixed-only with the same label counts.
                 If breaking pairs loses the gain but the control keeps it, pairing is the driver.
3. dose:         mixed-only plus ONE single-label training source at a time. Which source brings
                 the false positives back?
4. masking:      all-data training with the "openclaw" token removed, and with YAML frontmatter
                 stripped. Does removing the obvious shortcut fix all-data training?
5. bootstrap:    95% paired bootstrap CIs (2,000 resamples, class sizes fixed) for all-data vs
                 mixed-only, and exact Clopper-Pearson intervals for per-source rates.
"""

import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from skillshift.classical import fit_word_lr
from skillshift.data import load_split
from skillshift.metrics import bootstrap_ci, clopper_pearson, recall_at_fpr

MIXED_SOURCES = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}
SEEDS = (0, 1, 2)
FRONTMATTER = re.compile(r"\A\s*---\s*\n(.*?)\n---\s*\n", re.DOTALL)
NAME = re.compile(r"^name:\s*['\"]?([^'\"\n]+?)['\"]?\s*$", re.MULTILINE)


def skill_name(text):
    m = FRONTMATTER.match(text)
    if not m:
        return None
    n = NAME.search(m.group(1))
    return n.group(1).strip().lower() if n else None


def strip_frontmatter(text):
    return FRONTMATTER.sub("", text, count=1)


def drop_openclaw(text):
    return re.sub(r"openclaw", " ", text, flags=re.IGNORECASE)


def benign_fpr(y, s):
    return float((s[y == 0] >= 0.5).mean())


def pr_auc(y, s):
    return float(average_precision_score(y, s))


def recall_2pct(y, s):
    return recall_at_fpr(y, s, 0.02)


def summary(r):
    d = r["default"]
    return (f"n={r['train_n']:5d} FPR {d['benign_fpr']:.3f} recall {d['malicious_recall']:.3f} "
            f"PR-AUC {d['pr_auc']:.3f} R@2%FPR {d['recall_at_2pct_fpr']:.3f}")


def main() -> None:
    df = load_split("source_disjoint")
    df["name"] = df.text.map(skill_name)
    train_all, val_all, test = (df[df.split == s].copy() for s in ("train", "validation", "test"))
    mixed = train_all[train_all.source_name.isin(MIXED_SOURCES)]
    val_mixed = val_all[val_all.source_name.isin(MIXED_SOURCES)]
    out = {}

    # 1. pairing statistics inside the mixed-label training sources
    named = mixed.dropna(subset=["name"])
    by_key = named.groupby(["source_name", "name"]).label.agg(["min", "max", "size"])
    twin_keys = set(by_key[(by_key["min"] == 0) & (by_key["max"] == 1)].index)
    is_twin = mixed.apply(lambda r: (r.source_name, r["name"]) in twin_keys, axis=1)
    kept, added = [], []
    benign_by_key = {k: g.text.iloc[0] for k, g in named[named.label == 0].groupby(["source_name", "name"])}
    for _, r in mixed[is_twin & (mixed.label == 1)].iterrows():
        b_lines = [ln for ln in benign_by_key[(r.source_name, r["name"])].splitlines() if ln.strip()]
        m_lines = [ln for ln in r.text.splitlines() if ln.strip()]
        m_set, b_set = set(m_lines), set(b_lines)
        kept.append(np.mean([ln in m_set for ln in b_lines]) if b_lines else 0.0)
        added.append(sum(ln not in b_set for ln in m_lines))
    out["pairing"] = {
        "mixed_train_n": len(mixed),
        "by_source": mixed.source_name.value_counts().to_dict(),
        "malicious": int((mixed.label == 1).sum()),
        "malicious_with_benign_twin": int((is_twin & (mixed.label == 1)).sum()),
        "benign_with_malicious_twin": int((is_twin & (mixed.label == 0)).sum()),
        "twin_groups": len(twin_keys),
        "share_of_twin_lines_kept_median": round(float(np.median(kept)), 3),
        "lines_added_median": int(np.median(added)),
        "lines_added_p90": int(np.percentile(added, 90)),
    }
    print("pairing:", json.dumps(out["pairing"]))

    runs, scores = {}, {}

    def run(key, train, val, transform=None):
        res, s = fit_word_lr(train, val, test, transform)
        runs[key], scores[key] = res, s
        print(f"{key:34s} {summary(res)}", flush=True)

    run("all", train_all, val_all)
    run("mixed_only", mixed, val_mixed)

    # 2. break pairs: one side of every twin group, vs size-matched random subsets of mixed-only
    group = mixed.apply(lambda r: f"{r.source_name}|{r['name']}", axis=1)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        side = {g: int(rng.integers(0, 2)) for g in set(group[is_twin])}
        keep = ~is_twin | (mixed.label == group.map(side))
        one_side = mixed[keep]
        run(f"break_pairs_s{seed}", one_side, val_mixed)
        control = pd.concat([
            mixed[mixed.label == y].sample(int((one_side.label == y).sum()), random_state=seed)
            for y in (0, 1)
        ])
        run(f"size_matched_mixed_s{seed}", control, val_mixed)

    # 3. dose-response: add one single-label source at a time to mixed-only
    for src in sorted(set(train_all.source_name) - MIXED_SOURCES):
        extra = train_all[train_all.source_name == src]
        run(f"mixed_plus/{src}", pd.concat([mixed, extra]),
            val_all[val_all.source_name.isin(MIXED_SOURCES | {src})])

    # 4. masking the obvious shortcut in all-data training
    run("all_no_openclaw", train_all, val_all, drop_openclaw)
    run("all_no_frontmatter", train_all, val_all, strip_frontmatter)

    out["runs"] = runs
    for name, keys in [("break_pairs", "break_pairs_s"), ("size_matched_mixed", "size_matched_mixed_s")]:
        vals = [runs[f"{keys}{s}"]["default"] for s in SEEDS]
        out[f"{name}_mean"] = {m: round(float(np.mean([v[m] for v in vals])), 3)
                               for m in ("benign_fpr", "malicious_recall", "pr_auc", "recall_at_2pct_fpr")}

    # 5. confidence intervals for the headline comparison
    y = test.label.to_numpy()
    pair = {"mixed_only": scores["mixed_only"], "all": scores["all"]}
    out["bootstrap"] = {
        "benign_fpr_at_0.5": bootstrap_ci(y, pair, benign_fpr),
        "pr_auc": bootstrap_ci(y, pair, pr_auc),
        "recall_at_2pct_fpr_oracle": bootstrap_ci(y, pair, recall_2pct),
    }
    exact = {}
    for key in ("all", "mixed_only"):
        pred = scores[key] >= 0.5
        rows = {}
        for src, g in test.assign(pred=pred).groupby("source_name"):
            for lab, what in ((0, "benign_fpr"), (1, "malicious_recall")):
                sub = g[g.label == lab]
                if len(sub):
                    rows[f"{src}/{what}"] = clopper_pearson(int(sub.pred.sum()), len(sub)) + [len(sub)]
        exact[key] = rows
    out["exact_per_source"] = exact
    print("bootstrap:", json.dumps(out["bootstrap"]))

    with open("results/mechanism.json", "w") as f:
        json.dump(out, f, indent=2)
    test[["benchmark_id", "source_name", "label"]].assign(
        word_lr_all=scores["all"], word_lr_mixed_only=scores["mixed_only"]
    ).to_csv("results/classical_test_scores.csv", index=False)


if __name__ == "__main__":
    main()
