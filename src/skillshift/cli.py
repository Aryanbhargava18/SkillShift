"""Scan SKILL.md files from the command line.

    uv run skillshift-scan path/to/SKILL.md [more files or directories...]

Directories are searched for SKILL.md files. Exit code is 1 if anything is flagged,
so it can gate a CI job or a pre-install hook.
"""

import argparse
import json
import sys
from pathlib import Path

from skillshift.scorer import OnnxScorer


def find_skills(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(p.rglob("SKILL.md"))
        else:
            yield p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--model-dir", default="models/serve")
    ap.add_argument("--threshold", type=float, help="override the model's calibrated threshold")
    ap.add_argument("--json", action="store_true", help="one JSON object per file")
    args = ap.parse_args(argv)

    scorer = OnnxScorer(args.model_dir)
    threshold = args.threshold if args.threshold is not None else float(scorer.config["threshold"])
    flagged = 0
    for path in find_skills(args.paths):
        s = scorer.score(path.read_text(errors="replace"))
        verdict = "malicious" if s.probability >= threshold else "benign"
        flagged += verdict == "malicious"
        if args.json:
            print(json.dumps({"path": str(path), "verdict": verdict, "malicious_probability": round(s.probability, 4),
                              "truncated": s.truncated}))
        else:
            note = "  (only first part read)" if s.truncated else ""
            print(f"{verdict:9s} {s.probability:.3f}  {path}{note}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
