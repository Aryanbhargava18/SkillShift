"""Export the demo's two fitted detectors to demo/web/models.json for the static, client-side demo.

demo/web/index.html reimplements TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, norm="l2") plus
logistic regression in JavaScript. It needs only, per model, each vocabulary term's idf and coefficient
and the intercept. Run after scripts/build_demo_models.py:

    uv run python scripts/export_static_models.py
"""

import json
from pathlib import Path

import joblib

SRC = Path("demo/models.joblib")
DST = Path("demo/web/models.json")


def export(vec, model) -> dict:
    idf, coef = vec.idf_, model.coef_[0]
    terms = {t: [round(float(idf[c]), 5), round(float(coef[c]), 6)] for t, c in vec.vocabulary_.items()}
    return {"terms": terms, "intercept": float(model.intercept_[0])}


def main() -> None:
    models = joblib.load(SRC)
    out = {name: export(*models[name]) for name in ("all", "mix")}
    with DST.open("w") as f:
        json.dump(out, f)
    print(f"wrote {DST} ({DST.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
