"""Step 9: LoRA fine-tune a small instruct LLM as a skill classifier, on the same
source-disjoint protocol and eval harness as the ModernBERT run (scripts/train_transformer.py).

Run (small model, fits an 8 GB MPS machine):
    uv run python scripts/train_lora.py --protocol source_disjoint --variant mixed_only
Larger base model (needs a CUDA GPU; QLoRA/bitsandbytes does not run on Apple MPS):
    uv run python scripts/train_lora.py --model Qwen/Qwen2.5-1.5B --4bit    # on CUDA only

Writes the SAME schema as the transformer run so compare_models.py and make_tables.py pick it up:
    results/transformer/<run>.json         metrics on test (default + val-calibrated thresholds)
    results/transformer/<run>_scores.csv   per-skill validation/test scores
    models/<run>/                          LoRA adapter + tokenizer (gitignored)

Why this exists:
- Paper: a third model class (LoRA on a decoder LLM, as SkillScreen used) alongside TF-IDF and
  ModernBERT. If the shortcut survives here too, it is a property of the data, not the model.
- Portfolio: a self-contained "parameter-efficient fine-tune + evaluation harness" artifact, which
  is the concrete thing many LLM job postings ask for. The harness is skillshift.metrics + the
  source-disjoint split; LoRA only swaps the model.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from skillshift.data import load_split
from skillshift.metrics import per_source, report, threshold_for_fpr

MIXED_SOURCES = {"SkillTrustBench", "AgentTrap", "SkillTrojan"}
# q/k/v/o projections exist in Llama/Qwen/Mistral-style decoders and in ModernBERT's attention.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def score(model, tok, texts, device, max_len, batch=8):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(device)
        logits = model(**enc).logits.float()
        out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--protocol", default="source_disjoint")
    ap.add_argument("--variant", default="mixed_only", choices=["all", "mixed_only"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)  # LoRA tolerates a higher lr than full FT
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--4bit", dest="fourbit", action="store_true",
                    help="QLoRA 4-bit base (CUDA only; ignored on MPS/CPU)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    short = args.model.split("/")[-1].lower().replace(".", "")
    run = f"lora_{short}_{args.protocol}_{args.variant}_s{args.seed}"

    df = load_split(args.protocol)
    train, val, test = (df[df.split == s] for s in ("train", "validation", "test"))
    if args.variant == "mixed_only":
        train = train[train.source_name.isin(MIXED_SOURCES)]
        val = val[val.source_name.isin(MIXED_SOURCES)]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # decoder LLMs often ship without a pad token

    model_kwargs = {"num_labels": 2}
    if args.fourbit and device == "cuda":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    elif args.fourbit:
        print("--4bit ignored: bitsandbytes needs CUDA, not MPS/CPU. Using fp16 LoRA.")

    base = AutoModelForSequenceClassification.from_pretrained(args.model, **model_kwargs)
    base.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(base, lora).to(device)
    model.print_trainable_parameters()  # the PEFT selling point: <1% of params trained

    # Class-weighted loss, so benign skills (the minority) count as much as malicious ones.
    w = torch.tensor([len(train) / (2 * (train.label == c).sum()) for c in (0, 1)],
                     dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / (args.batch * args.accum)) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    texts, labels = train.text.tolist(), train.label.tolist()
    t0, step, running = time.time(), 0, 0.0
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(texts))
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            y = torch.tensor([labels[j] for j in idx], device=device)
            loss = loss_fn(model(**enc).logits, y) / args.accum
            loss.backward()
            running += loss.item() * args.accum
            if (i // args.batch + 1) % args.accum == 0 or i + args.batch >= len(order):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
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
        "run": run, "args": vars(args), "train_n": len(train),
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
