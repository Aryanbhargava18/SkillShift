"""Precompute the two word-TF-IDF + logistic-regression detectors the demo serves, and write them to
demo/models.joblib. Shipping the fitted models (rather than fitting on startup) lets the Hugging Face
Space start instantly with no dataset download. Re-run this if the demo's model choice changes.

    uv run python scripts/build_demo_models.py
"""

import joblib
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MIXED = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}


def fit(train):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30_000, sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    model.fit(vec.fit_transform(train.text), train.label)
    return vec, model


def main() -> None:
    d = snapshot_download("ProtectSkills/MaliciousSkillBench", repo_type="dataset",
                          local_dir="data/raw/msb")
    p = pd.read_parquet(f"{d}/primary.parquet")
    p["text"] = p.skill_text.fillna(p.public_skill_text).fillna("")
    sp = pd.read_parquet(f"{d}/splits/source_disjoint.parquet")
    df = sp[["benchmark_id", "split"]].merge(
        p[["benchmark_id", "text", "label", "source_name"]], on="benchmark_id")
    df["label"] = df.label.astype(int)
    train = df[df.split == "train"]
    models = {"all": fit(train), "mix": fit(train[train.source_name.isin(MIXED)])}
    joblib.dump(models, "demo/models.joblib", compress=3)
    print("wrote demo/models.joblib")


if __name__ == "__main__":
    main()
