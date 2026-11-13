"""Step 2: simple baselines on all three protocols.

Run: uv run python scripts/baselines.py
Writes results/baselines.json.

1. keyword: flags a skill if it contains any hand-picked suspicious pattern. No training.
2. tfidf_lr: word TF-IDF + logistic regression.
3. tfidf_svm: word TF-IDF + linear SVM, the paper's baseline (should reproduce ~0.93 random vs ~0.67 source-disjoint).
"""

import json
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, "src")
from skillshift.data import PROTOCOLS, load_split  # noqa: E402
from skillshift.metrics import report  # noqa: E402

# Things a malicious skill often does: download-and-run, read secrets, send data out,
# hide payloads, or override the agent's instructions.
KEYWORDS = [
    r"curl[^\n|]*\|\s*(ba)?sh", r"wget[^\n|]*\|\s*(ba)?sh", r"base64\s+(-d|--decode)",
    r"\beval\(", r"\bexec\(", r"~/\.ssh", r"id_rsa", r"\.aws/credentials", r"\.env\b",
    r"/etc/passwd", r"api[_-]?key", r"secret", r"token", r"webhook", r"exfiltrat",
    r"ignore (all |any )?(previous|prior) instructions", r"do not (tell|inform) the user",
    r"rm\s+-rf", r"nc\s+-e", r"reverse shell", r"chmod\s+\+x", r"crontab",
]
PATTERN = re.compile("|".join(KEYWORDS), re.IGNORECASE)


def keyword_scores(texts):
    return np.array([len(PATTERN.findall(t)) for t in texts], dtype=float)


def main() -> None:
    results = {}
    for protocol in PROTOCOLS:
        df = load_split(protocol)
        train, test = df[df.split == "train"], df[df.split == "test"]
        print(f"\n== {protocol}: train {len(train)}, test {len(test)} "
              f"(test benign {(test.label == 0).sum()}, malicious {(test.label == 1).sum()})")
        res = {}

        s = keyword_scores(test.text)
        res["keyword"] = report(test.label, (s > 0).astype(int), s)

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
        x_train, x_test = vec.fit_transform(train.text), vec.transform(test.text)

        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x_train, train.label)
        res["tfidf_lr"] = report(test.label, lr.predict(x_test), lr.predict_proba(x_test)[:, 1])

        svm = LinearSVC(class_weight="balanced").fit(x_train, train.label)
        res["tfidf_svm"] = report(test.label, svm.predict(x_test), svm.decision_function(x_test))

        for name, r in res.items():
            print(f"  {name:10s} {r}")
        results[protocol] = res

    with open("results/baselines.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
