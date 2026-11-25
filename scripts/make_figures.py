"""Generate paper/README figures from results/*.json. Run: uv run python scripts/make_figures.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
OUT = Path("paper")


def load(name):
    return json.loads((R / name).read_text())


def main() -> None:
    mech = load("mechanism.json")
    tr_all = load("transformer/modernbert_source_disjoint_all_s0.json")
    tr_mix = load("transformer/modernbert_source_disjoint_mixed_only_s0.json")
    lora_all = load("transformer/lora_all.json")
    lora_mix = load("transformer/lora_mixed_only.json")
    ev = load("adv_evasion.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    # Left: benign FPR by model and training set (the headline)
    models = ["word TF-IDF\n+ LR", "ModernBERT\n(encoder)"]
    all_fpr = [mech["runs"]["all"]["default"]["benign_fpr"] * 100,
               tr_all["default"]["benign_fpr"] * 100]
    mix_fpr = [mech["runs"]["mixed_only"]["default"]["benign_fpr"] * 100,
               tr_mix["default"]["benign_fpr"] * 100]
    if lora_all and lora_mix:
        models.append("Qwen2.5-0.5B\nQLoRA (decoder)")
        all_fpr.append(lora_all["default"]["benign_fpr"] * 100)
        mix_fpr.append(lora_mix["default"]["benign_fpr"] * 100)
    x = range(len(models))
    ax1.bar([i - 0.2 for i in x], all_fpr, 0.4, label="all sources", color="#c0392b")
    ax1.bar([i + 0.2 for i in x], mix_fpr, 0.4, label="mixed-label only", color="#2980b9")
    for i, v in zip(x, all_fpr):
        ax1.text(i - 0.2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    for i, v in zip(x, mix_fpr):
        ax1.text(i + 0.2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax1.set_xticks(list(x)); ax1.set_xticklabels(models)
    ax1.set_ylabel("benign false-positive rate (%)")
    ax1.set_title("Over-blocking on unseen sources")
    ax1.set_ylim(0, 100); ax1.legend(frameon=False, fontsize=8)

    # Right: good-word attack dose-response (all-data model)
    ks = ev["ks"]; dr = ev["dose_response"]["all"]
    ax2.plot(ks, [v * 100 for v in dr["benign"]], "o-", color="#c0392b", label="source-style tokens")
    ax2.plot(ks, [v * 100 for v in dr["random"]], "s--", color="#7f8c8d", label="random tokens")
    ax2.set_xlabel("tokens appended (k)")
    ax2.set_ylabel("evasion success rate (%)")
    ax2.set_title("Good-word attack (all-data model)")
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_results.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'fig_results.png'}")


if __name__ == "__main__":
    main()
