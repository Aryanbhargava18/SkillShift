"""Print the README's result tables from results/*.json, so no number is typed by hand.

Run: uv run python scripts/make_tables.py
Sections whose result files don't exist yet are skipped.
"""

import json
from pathlib import Path

R = Path("results")


def load(path):
    p = R / path
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return f"{100 * x:.1f}%"


def row(name, m):
    return (f"| {name} | {m['macro_f1']:.3f} | {pct(m['benign_fpr'])} | {pct(m['malicious_recall'])} | "
            f"{m['pr_auc']:.3f} | {pct(m['recall_at_2pct_fpr'])} |")


HEADER = ("| model | macro-F1 | benign wrongly flagged | malicious caught | PR-AUC | recall @ 2% FPR* |\n"
          "|---|---|---|---|---|---|")


def main() -> None:
    base = load("baselines.json")
    if base:
        print("### Baselines across protocols (macro-F1 / benign wrongly flagged)\n")
        print("| model | random | structural-disjoint | source-disjoint |\n|---|---|---|---|")
        for m in ("keyword", "tfidf_lr", "tfidf_svm"):
            cells = [f"{base[p][m]['macro_f1']:.3f} / {pct(base[p][m]['benign_fpr'])}"
                     for p in ("random", "m_structural_disjoint", "source_disjoint")]
            print(f"| {m} | " + " | ".join(cells) + " |")
        print()

    probe = load("shortcut_probe.json")
    if probe:
        s = probe["source_id"]
        print(f"Source identification from text: accuracy {pct(s['accuracy'])}, balanced accuracy "
              f"{pct(s['balanced_accuracy'])} over {s['n_sources']} sources (chance {pct(s['chance_balanced_accuracy'])}).\n")

    rob = load("robust_classical.json")
    tr = {p.stem: json.loads(p.read_text()) for p in sorted((R / "transformer").glob("*.json"))}
    if rob or tr:
        print("### Source-disjoint test (unseen sources), default threshold\n")
        print(HEADER)
        if rob:
            for k in ("all/word_svm", "all/char_lr", "group_balanced/char_svm", "mixed_only/word_lr", "mixed_only/char_lr"):
                print(row(f"TF-IDF {k}", rob[k]["default"]))
            for feats in ("word_lr", "char_lr"):
                ms = [rob[f"size_matched_s{i}/{feats}"]["default"] for i in range(3)]
                avg = {k: sum(m[k] for m in ms) / 3 for k in ms[0]}
                print(row(f"TF-IDF size_matched/{feats} (mean of 3 seeds)", avg))
        for name, r in tr.items():
            if "source_disjoint" in name:
                print(row(name, r["default"]))
        print("\n*recall @ 2% FPR picks the threshold on the test set itself: a ceiling, not a deployable number.\n")

        print("### Source-disjoint test with the threshold fixed on validation (2% FPR there)\n")
        print("| model | benign wrongly flagged (test) | malicious caught (test) |\n|---|---|---|")
        if rob:
            for k in ("all/word_svm", "all/char_lr", "mixed_only/word_lr", "mixed_only/char_lr"):
                v = rob[k]["val_2pct"]
                print(f"| TF-IDF {k} | {pct(v['benign_fpr'])} | {pct(v['malicious_recall'])} |")
        for name, r in tr.items():
            if "source_disjoint" in name:
                v = r["val_2pct"]
                print(f"| {name} | {pct(v['benign_fpr'])} | {pct(v['malicious_recall'])} |")
        print()

    for name, r in tr.items():
        if "random" in name:
            print(f"In-distribution check, {name}: macro-F1 {r['default']['macro_f1']:.3f}, "
                  f"benign wrongly flagged {pct(r['default']['benign_fpr'])}.\n")

    for p in sorted(R.glob("onnx_*.json")):
        o = json.loads(p.read_text())
        a, lat = o["agreement_on_source_disjoint_test"], o["latency_single_request_cpu"]
        print(f"### Inference ({o['run']}, CPU, {o['threads']} threads, single request)\n")
        print("| format | size | p50 latency | p95 latency |\n|---|---|---|---|")
        print(f"| PyTorch fp32 | - | {lat['torch_fp32']['p50_ms']} ms | {lat['torch_fp32']['p95_ms']} ms |")
        print(f"| ONNX fp32 | {o['size_mb']['onnx_fp32']} MB | {lat['onnx_fp32']['p50_ms']} ms | {lat['onnx_fp32']['p95_ms']} ms |")
        print(f"| ONNX int8 | {o['size_mb']['onnx_int8']} MB | {lat['onnx_int8']['p50_ms']} ms | {lat['onnx_int8']['p95_ms']} ms |")
        print(f"\nint8 vs PyTorch on the {a['torch']['macro_f1']} -> {a['onnx_int8']['macro_f1']} macro-F1 test set: "
              f"max probability difference {a['max_abs_diff_int8_vs_torch']}, "
              f"{a['verdict_flips_int8_vs_torch']} verdicts flipped, tokenizer mismatches {a['tokenizer_mismatches_of_300']}/300.\n")

    for p in sorted(R.glob("chunked_*.json")):
        c = json.loads(p.read_text())
        print(f"### Head-only vs whole-skill scoring ({c['run']}, threshold fixed on validation)\n")
        print("| scoring | benign wrongly flagged | malicious caught | PR-AUC |\n|---|---|---|---|")
        for agg, label in (("head", "first 512 tokens"), ("max", "max over windows (up to ~3k tokens)")):
            v = c[agg]["val_2pct"]
            print(f"| {label} | {pct(v['benign_fpr'])} | {pct(v['malicious_recall'])} | {v['pr_auc']:.3f} |")
        print()


if __name__ == "__main__":
    main()
