"""Step 6: export a fine-tuned model to ONNX, quantise it to int8, and check what that costs.

Run: uv run python scripts/export_onnx.py --run modernbert_source_disjoint_mixed_only_s0

Steps:
1. Export the PyTorch model to ONNX (fp32), with dynamic batch and sequence length.
2. Dynamic int8 quantisation: weights stored as int8, activations quantised on the fly.
   About 4x smaller; usually faster on CPU.
3. Check the int8 model still agrees with PyTorch on the source-disjoint test set
   (metric change, max probability difference), and that our standalone tokenizer gives
   exactly the same token ids as the Hugging Face one.
4. Measure single-request CPU latency for PyTorch fp32, ONNX fp32 and ONNX int8.
5. Write models/serve/ (int8 model + tokenizer + config) for the service.

Writes results/onnx_<run>.json.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from skillshift.data import load_split
from skillshift.metrics import report
from skillshift.scorer import OnnxScorer


def mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 1)


def latency(fn, texts):
    for t in texts[:5]:  # warm-up
        fn(t)
    times = []
    for t in texts:
        start = time.perf_counter()
        fn(t)
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": round(float(np.percentile(times, 50)), 1),
            "p95_ms": round(float(np.percentile(times, 95)), 1),
            "mean_ms": round(float(np.mean(times)), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--latency-n", type=int, default=200)
    args = ap.parse_args()

    src = Path("models") / args.run
    out = Path("models") / f"{args.run}_onnx"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(f"results/transformer/{args.run}.json").read_text())
    max_len = meta["args"]["max_len"]
    torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()

    # 1. Export.
    dummy = tok(["export example"] * 2, padding="max_length", max_length=16, return_tensors="pt")
    fp32 = out / "model.fp32.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    # 2. Quantise.
    int8 = out / "model.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    (out / "skillshift.json").write_text(json.dumps({
        "model_version": args.run,
        "threshold": meta["val_2pct_threshold"],
        "max_length": max_len,
    }, indent=2))

    # 3. Agreement on the source-disjoint test set.
    df = load_split("source_disjoint")
    test = df[df.split == "test"]
    texts = test.text.tolist()

    @torch.no_grad()
    def torch_prob(text):
        enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
        return torch.softmax(model(**enc).logits, -1)[0, 1].item()

    s_fp32, s_int8 = OnnxScorer(out, "model.fp32.onnx", args.threads), OnnxScorer(out, "model.onnx", args.threads)
    mismatched_ids = sum(
        s_int8.tokenizer.encode(t).ids != tok(t, truncation=True, max_length=max_len)["input_ids"]
        for t in texts[:300]
    )
    p_torch = np.array([torch_prob(t) for t in texts])
    p_fp32 = np.array([s_fp32.score(t).probability for t in texts])
    p_int8 = np.array([s_int8.score(t).probability for t in texts])
    t = meta["val_2pct_threshold"]
    agreement = {
        "tokenizer_mismatches_of_300": int(mismatched_ids),
        "max_abs_diff_fp32_vs_torch": round(float(np.abs(p_fp32 - p_torch).max()), 5),
        "max_abs_diff_int8_vs_torch": round(float(np.abs(p_int8 - p_torch).max()), 5),
        "verdict_flips_int8_vs_torch": int(((p_int8 >= t) != (p_torch >= t)).sum()),
        "torch": report(test.label, (p_torch >= t).astype(int), p_torch),
        "onnx_int8": report(test.label, (p_int8 >= t).astype(int), p_int8),
    }

    # 4. Latency, one request at a time (what the service does).
    rng = np.random.default_rng(0)
    sample = [texts[i] for i in rng.choice(len(texts), args.latency_n, replace=False)]
    result = {
        "run": args.run, "threads": args.threads, "max_length": max_len,
        "size_mb": {"onnx_fp32": mb(fp32), "onnx_int8": mb(int8)},
        "agreement_on_source_disjoint_test": agreement,
        "latency_single_request_cpu": {
            "torch_fp32": latency(torch_prob, sample),
            "onnx_fp32": latency(s_fp32.score, sample),
            "onnx_int8": latency(s_int8.score, sample),
        },
    }
    print(json.dumps(result, indent=1))
    Path(f"results/onnx_{args.run}.json").write_text(json.dumps(result, indent=2))

    # 5. Serving directory.
    serve = Path("models/serve")
    serve.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "tokenizer.json", "skillshift.json"):
        shutil.copy(out / name, serve / name)


if __name__ == "__main__":
    main()
