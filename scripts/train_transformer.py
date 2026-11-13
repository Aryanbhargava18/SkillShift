"""Step 5: fine-tune ModernBERT-base to classify skills as malicious (1) or benign (0).

Run (Apple Silicon or CUDA):
    uv run python scripts/train_transformer.py --protocol source_disjoint --variant mixed_only

Writes:
    results/transformer/<run>.json         metrics on test (default and val-calibrated thresholds)
    results/transformer/<run>_scores.csv   per-skill validation/test scores, for later analysis
    models/<run>/                          fine-tuned weights + tokenizer (gitignored)

Design choices, kept simple on purpose:
- Input: the first 512 tokens of the skill (about 2,000 characters). Median skill is ~1,250 tokens,
  so long skills are cut; scripts/eval_chunked.py checks how much that costs.
- Loss: cross-entropy with class weights, so benign skills (the minority) count as much as malicious.
- Optimiser: AdamW, lr 3e-5, linear warmup (10%) then linear decay, 2 epochs, effective batch 16.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from skillshift.data import load_split
from skillshift.metrics import per_source, report, threshold_for_fpr

MODEL = "answerdotai/ModernBERT-base"
MIXED_SOURCES = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def score(model, tok, texts, device, max_len, batch=8):
    """Return P(malicious) for each text."""
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_step(model, opt, tok, loss_fn, texts, labels, args, device, retries=3):
    """Forward + backward on one batch; returns the loss, or None if it never fit in memory.

    On an out-of-memory error the backward pass may have stopped halfway, leaving partial
    gradients. So we clear all gradients (this drops the current accumulation window, at most
    `accum` batches), free cached GPU memory, wait, and retry.
    """
    for attempt in range(retries):
        try:
            enc = tok(texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to(device)
            y = torch.tensor(labels, device=device)
            loss = loss_fn(model(**enc).logits, y) / args.accum
            loss.backward()
            return loss.item()
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()
            time.sleep(5 * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="source_disjoint")
    ap.add_argument("--variant", default="all", choices=["all", "mixed_only"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training skills")
    args = ap.parse_args()

    run = f"modernbert_{args.protocol}_{args.variant}_s{args.seed}"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device()

    df = load_split(args.protocol)
    train, val, test = (df[df.split == s] for s in ("train", "validation", "test"))
    if args.variant == "mixed_only":
        train = train[train.source_name.isin(MIXED_SOURCES)]
        val = val[val.source_name.isin(MIXED_SOURCES)]
    if args.limit:
        train = train.sample(args.limit, random_state=args.seed)
    print(f"{run}: device {device}, train {len(train)}, val {len(val)}, test {len(test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)

    # Class weights: inverse frequency, so each class contributes equally to the loss.
    counts = np.bincount(train.label, minlength=2)
    weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train) / (args.batch * args.accum)) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    texts, labels = train.text.tolist(), train.label.to_numpy()
    t0, step, skipped = time.time(), 0, 0
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(texts))
        running = 0.0
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            loss = train_step(model, opt, tok, loss_fn, [texts[j] for j in idx], labels[idx], args, device)
            if loss is None:
                skipped += len(idx)
                print(f"skipped batch after repeated out-of-memory errors ({skipped} skills so far)", flush=True)
            else:
                running += loss
            if (i // args.batch + 1) % args.accum == 0 or i + args.batch >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if device == "mps" and step % 50 == 0:
                    torch.mps.empty_cache()
                if step % 25 == 0:
                    print(f"epoch {epoch} step {step}/{steps} loss {running / 25:.4f} "
                          f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
                    running = 0.0

    s_val = score(model, tok, val.text.tolist(), device, args.max_len)
    s_test = score(model, tok, test.text.tolist(), device, args.max_len)
    t = threshold_for_fpr(val.label, s_val, 0.02)
    result = {
        "run": run, "args": vars(args), "train_n": len(train), "skipped_oom": skipped,
        "train_minutes": round((time.time() - t0) / 60, 1),
        "val": report(val.label, (s_val >= 0.5).astype(int), s_val),
        "default": report(test.label, (s_test >= 0.5).astype(int), s_test),
        "default_per_source": per_source(test, (s_test >= 0.5).astype(int)),
        "val_2pct_threshold": t,
        "val_2pct": report(test.label, (s_test >= t).astype(int), s_test),
        "val_2pct_per_source": per_source(test, (s_test >= t).astype(int)),
    }
    print(json.dumps({k: result[k] for k in ("default", "val_2pct")}, indent=1), flush=True)

    os.makedirs("results/transformer", exist_ok=True)
    with open(f"results/transformer/{run}.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.concat([
        val.assign(score=s_val)[["benchmark_id", "split", "source_name", "label", "score"]],
        test.assign(score=s_test)[["benchmark_id", "split", "source_name", "label", "score"]],
    ]).to_csv(f"results/transformer/{run}_scores.csv", index=False)
    model.save_pretrained(f"models/{run}"); tok.save_pretrained(f"models/{run}")


if __name__ == "__main__":
    main()
