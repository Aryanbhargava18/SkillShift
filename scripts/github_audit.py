"""Step 11: real-world audit. How many real GitHub SKILL.md files does each detector flag?

Run: uv run python scripts/github_audit.py   (needs the gh CLI authenticated)
Collects public SKILL.md files from GitHub (once, into data/github_wild/) by enumerating each
candidate repo's tree, then scores them with the all-data and mixed-only word-LR detectors at their
deployment thresholds. These wild skills are UNLABELED but overwhelmingly benign (real published
skills), so the fraction a detector flags as malicious is a proxy for its real-world over-blocking.
Writes results/github_audit.json.
"""

# ruff: noqa: BLE001, S112
import base64
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from skillshift.classical import word_vec
from skillshift.data import load_split
from skillshift.metrics import threshold_for_fpr

WILD = Path("data/github_wild")
MIXED = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}
MAX_FILES = 400
MAX_PER_REPO = 25


def gh(args):
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=False).stdout


def candidate_repos():
    repos = []
    for q in ("path:SKILL.md", "openclaw SKILL.md", "agent skill SKILL.md"):
        for page in range(1, 4):
            out = gh(["api", "-X", "GET", "search/code", "-f", f"q={q}",
                      "-f", "per_page=100", "-f", f"page={page}"])
            try:
                items = json.loads(out).get("items", [])
            except json.JSONDecodeError:
                break
            if not items:
                break
            for it in items:
                r = it["repository"]["full_name"]
                if r not in repos:
                    repos.append(r)
    return repos


def collect():
    WILD.mkdir(parents=True, exist_ok=True)
    if list(WILD.glob("*.md")):
        return
    n = 0
    for repo in candidate_repos():
        if n >= MAX_FILES:
            break
        try:
            br = json.loads(gh(["api", f"repos/{repo}"])).get("default_branch", "main")
            tree = json.loads(gh(["api", f"repos/{repo}/git/trees/{br}?recursive=1"])).get("tree", [])
        except json.JSONDecodeError:
            continue
        blobs = [t for t in tree if t.get("type") == "blob"
                 and t["path"].split("/")[-1].lower() == "skill.md"]
        per = 0
        for t in blobs:
            if per >= MAX_PER_REPO or n >= MAX_FILES:
                break
            try:
                b = json.loads(gh(["api", f"repos/{repo}/git/blobs/{t['sha']}"]))
                text = base64.b64decode(b["content"]).decode("utf-8", "ignore")
            except Exception:
                continue
            if len(text) < 50:
                continue
            (WILD / f"{n:04d}.md").write_text(text)
            n += 1
            per += 1
    print(f"collected {n} wild SKILL.md files")


def fit(train):
    vec = word_vec()
    model = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0)
    model.fit(vec.fit_transform(train.text), train.label)
    return vec, model


def main() -> None:
    collect()
    wild = [p.read_text() for p in sorted(WILD.glob("*.md"))]
    openclaw = sum("openclaw" in t.lower() for t in wild)
    df = load_split("source_disjoint")
    train, val = df[df.split == "train"], df[df.split == "validation"]
    out = {"n_wild": len(wild), "wild_with_openclaw": openclaw, "models": {}}
    for name, tr, va in [("all", train, val),
                         ("mixed_only", train[train.source_name.isin(MIXED)],
                          val[val.source_name.isin(MIXED)])]:
        vec, model = fit(tr)
        thr = threshold_for_fpr(va.label, model.predict_proba(vec.transform(va.text))[:, 1], 0.02)
        s = model.predict_proba(vec.transform(wild))[:, 1]
        out["models"][name] = {
            "flagged_at_0.5": round(float((s >= 0.5).mean()), 3),
            "flagged_at_deploy_threshold": round(float((s >= thr).mean()), 3),
            "median_malicious_score": round(float(np.median(s)), 3),
        }
        print(f"{name:11s} flags {out['models'][name]['flagged_at_0.5']*100:.0f}% @0.5, "
              f"{out['models'][name]['flagged_at_deploy_threshold']*100:.0f}% @deploy-threshold")
    with open("results/github_audit.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
