"""Step 1: look at MaliciousSkillBench before modelling anything.

Run: uv run python scripts/inspect_data.py
Downloads nothing; expects data/raw/msb (see README for the download command).
"""

import pandas as pd

DATA = "data/raw/msb/"


def main() -> None:
    primary = pd.read_parquet(DATA + "primary.parquet")
    print("rows, columns:", primary.shape)
    print("labels (1 = malicious):", primary.label.value_counts().to_dict())

    # Which benchmark each sample came from, and its label mix.
    # Most sources are ALL malicious or ALL benign: this is the shortcut a model can learn.
    print("\nsource x label:")
    print(pd.crosstab(primary.source_name, primary.label).to_string())

    chars = primary.skill_text.fillna("").str.len()
    print(
        "\nskill_text length in chars: median",
        int(chars.median()),
        "p90",
        int(chars.quantile(0.9)),
        "p99",
        int(chars.quantile(0.99)),
    )

    for protocol in ["random", "m_structural_disjoint", "source_disjoint"]:
        split = pd.read_parquet(DATA + f"splits/{protocol}.parquet")
        print(f"\nprotocol {protocol}:")
        print(pd.crosstab(split.split, split.label).to_string())


if __name__ == "__main__":
    main()
