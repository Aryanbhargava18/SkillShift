# SkillShift

**Can a text classifier detect malicious agent skills (`SKILL.md`) from sources it has never seen — or does it just learn to recognise the source?**

On [MaliciousSkillBench](https://huggingface.co/datasets/ProtectSkills/MaliciousSkillBench), detectors that look strong on a random split wrongly flag **two-thirds of benign skills from sources they were not trained on**. This repo shows *why*, shows a fix, and shows the same bias is an attack surface. On **400 real GitHub skills**, the merged-data detector flags **30%** as malicious while the fix flags **5%**. Full writeup: [paper/skillshift.pdf](paper/skillshift.pdf). Working notes: [NOTES.md](NOTES.md).

## Findings

**1. The gap is real and reproduces the benchmark.** Benign false-positive rate on unseen-source test skills (95% bootstrap CI):

| model | benign FPR | PR-AUC |
|---|---|---|
| word TF-IDF + LR, all data | 65% [61–69] | 0.79 |
| **word TF-IDF + LR, mixed-label sources only** | **3.7% [2.2–5.3]** | **0.93** |
| ModernBERT-base, all data | 79% [76–82] | 0.66 |
| ModernBERT-base, mixed-label sources only | 20% [17–23] | 0.79 |

**2. The model learns the source, not the label.** A TF-IDF probe predicts a skill's *source benchmark* from its text with **91.7% accuracy** (chance 10%). Tokens like `openclaw` act as "benign" cues only because of how the benchmark's sources were mixed.

**3. The fix: train only on sources that contain both labels.** This cuts benign FPR from 65% to 3.7% and raises PR-AUC to 0.93 — and a **size-matched random control stays at 58%**, so it is *which* data, not *less* data. A simple linear model with the right data beats a fine-tuned transformer with the wrong data.

**4. What drives it (ablations with controls).** It is **label-mixing, not minimal-pair construction**: breaking the benign/malicious twin pairs moves FPR only 3.7%→9.7% (≈ the 6.5% size-matched control, ≪ 65% all-data). **MalSkillBench** is the specific culprit — adding its single-label malicious skills alone drags FPR back to 61%. Masking the `openclaw` token (64%) or stripping YAML frontmatter (60%) does **not** fix it; the shortcut is distributed.

**5. The shortcut is an attack surface, and the fix helps there too.** Appending **8** of the detector's own benign source-style tokens to a caught malicious skill evades it **17%** of the time — vs **5%** for 8 random tokens (a good-word attack, Lowd & Meek 2005). The mixed-only fix lowers this to **7%**. One bias → over-blocking benign *and* evadable malicious → one data fix moves both.

## Artifacts
- `src/skillshift/` — TF-IDF baselines, metrics with bootstrap + Clopper-Pearson CIs, an int8 **ONNX scanner**, a **FastAPI** scan service with OpenTelemetry spans, and a `skillshift-scan` CLI.
- `scripts/` — every experiment (steps 1–10); all paper numbers are generated from `results/*.json` (`make_paper_numbers.py`), none typed by hand.
- `Dockerfile` — torch-free inference image (ONNX Runtime + tokenizer + FastAPI).

## Reproduce
```bash
uv sync
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ProtectSkills/MaliciousSkillBench', repo_type='dataset', local_dir='data/raw/msb')"
uv run python scripts/baselines.py        # steps 1-2: baselines across the three splits
uv run python scripts/shortcut_probe.py   # step 3: source is predictable from text
uv run python scripts/robust_classical.py # step 4: the mixed-only fix + controls
./scripts/run_transformer_suite.sh        # steps 5-6: ModernBERT (both training sets)
uv run python scripts/mechanism.py        # step 7: pairing / dose-response / masking
uv run python scripts/compare_models.py   # step 8: all models, paired bootstrap CIs
uv run python scripts/adv_evasion.py      # step 10: the good-word attack, dose-response
```
