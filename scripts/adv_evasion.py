"""Step 10: is the source shortcut a usable evasion vector, cleanly separated from padding?

Run: uv run python scripts/adv_evasion.py   (writes results/adv_evasion.json)

A good-word attack (Lowd & Meek 2005) appends benign-indicative tokens to a malicious skill to pull
its score below threshold. The honest question is whether the SOURCE-STYLE tokens evade more than the
same number of RANDOM tokens -- i.e. is there a source-specific effect, or is it just that appending
text dilutes a TF-IDF vector? We answer with a dose-response: at each injection size k, compare
appending the model's top-k benign features vs k random vocabulary tokens. The confound-free signal is
ASR(benign, k) - ASR(random, k) at small k, before padding dominates.

We also report both the vulnerable all-data model and the mixed-only "fixed" model, to test whether
the false-positive fix transfers to evasion-robustness (it does not have to).
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from skillshift.classical import word_vec
from skillshift.data import load_split
from skillshift.metrics import threshold_for_fpr

MIXED_SOURCES = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}
KS = [1, 2, 3, 5, 8, 13, 21, 40]
HEADLINE_K = 8          # small enough that random padding barely moves the vector
SEED = 0


def fit(train):
    vec = word_vec()
    model = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0)
    model.fit(vec.fit_transform(train.text), train.label)
    return vec, model


def pmal(vec, model, texts):
    return model.predict_proba(vec.transform(texts))[:, 1]


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load_split("source_disjoint")
    train, val, test = (df[df.split == s] for s in ("train", "validation", "test"))
    mixed_tr = train[train.source_name.isin(MIXED_SOURCES)]
    mixed_va = val[val.source_name.isin(MIXED_SOURCES)]
    mal = test[test.label == 1]

    vec_all, m_all = fit(train)
    vec_mix, m_mix = fit(mixed_tr)
    feats = np.asarray(vec_all.get_feature_names_out())
    benign_ranked = feats[np.argsort(m_all.coef_[0])].tolist()        # most-benign first
    rand_pool = rng.permutation(feats).tolist()

    def block(tokens):
        return "\n\n## metadata\n" + " ".join(tokens) + "\n"

    def asr(vec, model, thr, flagged, base_scores, tokens):
        atk = pmal(vec, model, (mal.text + block(tokens)).tolist())
        return float(((atk < thr) & flagged).mean()) if flagged.any() else 0.0

    out = {"headline_k": HEADLINE_K, "ks": KS, "dose_response": {}, "models": {}}
    for mname, (vec, model, vsplit) in {
        "all": (vec_all, m_all, val), "mixed_only": (vec_mix, m_mix, mixed_va),
    }.items():
        thr = threshold_for_fpr(vsplit.label, pmal(vec, model, vsplit.text), 0.02)
        base = pmal(vec, model, mal.text)
        flagged = base >= thr          # malicious skills this model catches (the attack's targets)
        n = int(flagged.sum())
        curve = {"benign": [], "random": []}
        for k in KS:
            curve["benign"].append(round(asr(vec, model, thr, flagged, base, benign_ranked[:k]), 3))
            curve["random"].append(round(asr(vec, model, thr, flagged, base, rand_pool[:k]), 3))
        out["dose_response"][mname] = curve
        hi = KS.index(HEADLINE_K)
        out["models"][mname] = {
            "threshold": round(float(thr), 4), "malicious_flagged": n,
            "asr_benign_at_k": curve["benign"][hi], "asr_random_at_k": curve["random"][hi],
            "source_specific_gap": round(curve["benign"][hi] - curve["random"][hi], 3),
        }
        print(f"{mname:11s} n={n:4d}  benign@{HEADLINE_K}={curve['benign'][hi]}  "
              f"random@{HEADLINE_K}={curve['random'][hi]}  gap={out['models'][mname]['source_specific_gap']}",
              flush=True)
    print("dose-response (all model):", out["dose_response"]["all"])

    with open("results/adv_evasion.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
