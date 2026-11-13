"""Load MaliciousSkillBench texts joined to a frozen split manifest."""

import pandas as pd

DATA = "data/raw/msb/"
PROTOCOLS = ["random", "m_structural_disjoint", "source_disjoint"]


def load_split(protocol: str) -> pd.DataFrame:
    """Return one row per skill: text, label (int), source, split."""
    primary = pd.read_parquet(DATA + "primary.parquet")
    # Dataset README: use exact text, fall back to the sanitised text for the 5 withheld records.
    primary["text"] = primary.skill_text.fillna(primary.public_skill_text).fillna("")
    split = pd.read_parquet(DATA + f"splits/{protocol}.parquet")
    df = split[["benchmark_id", "split"]].merge(
        primary[["benchmark_id", "text", "label", "source_name"]], on="benchmark_id"
    )
    df["label"] = df.label.astype(int)
    return df[df.split != "excluded"].reset_index(drop=True)
