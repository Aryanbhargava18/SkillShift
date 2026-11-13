"""Does reading the whole skill help, or is the first 512 tokens enough?

Run: uv run python scripts/eval_chunked.py --run modernbert_source_disjoint_mixed_only_s0

The model was trained on the first 512 tokens. Here each skill is split into overlapping
512-token windows (stride 384, at most 8 windows ~ 3,000 tokens) and scored per window.
A skill's score is the MAX over windows: one malicious section should be enough to flag it.
Compared against the head-only score on the same validation/test skills.
Writes results/chunked_<run>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from skillshift.data import load_split
from skillshift.metrics import per_source, report, threshold_for_fpr


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def window_scores(model, tok, text, dev, max_len=512, stride=384, max_windows=8):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    body = max_len - 2  # room for [CLS] and [SEP]
    starts = list(range(0, max(len(ids) - body, 0) + 1, stride))[:max_windows] or [0]
    chunks = [[tok.cls_token_id] + ids[s:s + body] + [tok.sep_token_id] for s in starts]
    width = max(len(c) for c in chunks)
    input_ids = torch.tensor([c + [tok.pad_token_id] * (width - len(c)) for c in chunks], device=dev)
    mask = torch.tensor([[1] * len(c) + [0] * (width - len(c)) for c in chunks], device=dev)
    return torch.softmax(model(input_ids=input_ids, attention_mask=mask).logits.float(), -1)[:, 1].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    meta = json.loads(Path(f"results/transformer/{args.run}.json").read_text())
    dev = device()
    tok = AutoTokenizer.from_pretrained(f"models/{args.run}")
    model = AutoModelForSequenceClassification.from_pretrained(f"models/{args.run}").to(dev).eval()

    df = load_split(meta["args"]["protocol"])
    val, test = df[df.split == "validation"], df[df.split == "test"]
    if meta["args"]["variant"] == "mixed_only":
        val = val[val.source_name.isin({"SkillTrustBench", "AgentTrap", "SkillTrojan"})]

    result = {"run": args.run}
    scores = {}
    for name, part in (("val", val), ("test", test)):
        w = [window_scores(model, tok, t, dev) for t in part.text]
        scores[name] = {"head": np.array([x[0] for x in w]), "max": np.array([x.max() for x in w]),
                        "n_windows": np.array([len(x) for x in w])}
    result["test_skills_with_more_than_one_window"] = int((scores["test"]["n_windows"] > 1).sum())
    for agg in ("head", "max"):
        t = threshold_for_fpr(val.label, scores["val"][agg], 0.02)
        s = scores["test"][agg]
        result[agg] = {
            "default": report(test.label, (s >= 0.5).astype(int), s),
            "val_2pct": report(test.label, (s >= t).astype(int), s),
            "val_2pct_per_source": per_source(test, (s >= t).astype(int)),
        }
    print(json.dumps({k: result[k] for k in ("head", "max")}, indent=1))
    Path(f"results/chunked_{args.run}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
