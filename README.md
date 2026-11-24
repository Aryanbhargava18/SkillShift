# SkillShift — diagnosing and fixing source shortcuts in malicious agent-skill detection

**Can a text classifier detect malicious agent skills (`SKILL.md`) from sources it has never seen — or does it just learn to recognise the source?** A short, end-to-end ML-security research project: diagnosis, a tested fix, mechanism ablations, an adversarial attack, and a shipped scanner.

On [MaliciousSkillBench](https://huggingface.co/datasets/ProtectSkills/MaliciousSkillBench) (9,740 skills), detectors that look strong on a random split wrongly flag **two-thirds of benign skills from sources they were not trained on**. This repo shows *why*, shows a fix, and shows the same bias is an attack surface. On **400 real GitHub skills**, the merged-data detector flags **30%** as malicious while the fix flags **5%**. Full writeup: [paper/skillshift.pdf](paper/skillshift.pdf).

![Benign false-positive rate by model, and the good-word attack dose-response](paper/fig_results.png)

## What this project demonstrates
- **LLM fine-tuning** — full fine-tuning of **ModernBERT**, and **parameter-efficient fine-tuning (LoRA / QLoRA, 4-bit) of Qwen2.5** via **PEFT** (`scripts/train_lora.py`, `docs/RUN_ON_GPU.md`).
- **Evaluation harness & statistical rigor** — an out-of-distribution (source-disjoint) evaluation protocol with **paired bootstrap confidence intervals** and **exact Clopper-Pearson** intervals for small groups; every paper number is generated from `results/*.json`, none typed by hand.
- **Adversarial ML / AI security** — a **good-word evasion attack** (dose-response, confound-controlled) against a learned detector; analysis of **shortcut learning / dataset bias / spurious correlations**; **red-team**-style robustness evaluation.
- **Model optimization & serving (MLOps)** — **ONNX** export with **int8 quantization**, a **FastAPI** inference service instrumented with **OpenTelemetry** distributed tracing, a `skillshift-scan` **CLI**, and a torch-free **Docker** image (built + healthchecked).
- **Stack** — PyTorch, Hugging Face Transformers & Hub, scikit-learn, pandas, ONNX Runtime, uv, pytest, GitHub Actions CI.

## Findings

**1. The gap reproduces, across model classes.** Benign false-positive rate (FPR) and malicious recall on unseen-source test skills (95% bootstrap CI on FPR; recall at the default 0.5 threshold):

| model | benign FPR | malicious recall | PR-AUC |
|---|---|---|---|
| word TF-IDF + Logistic Regression, all data | 65% [61–69] | 96% | 0.79 |
| **word TF-IDF + LR, mixed-label sources only** | **3.7% [2.2–5.3]** | 50% | **0.93** |
| ModernBERT-base, all data | 79% [76–82] | 98% | 0.66 |
| ModernBERT-base, mixed-label sources only | 20% [17–23] | 53% | 0.79 |

The fix **trades malicious recall at a fixed threshold** for the FPR drop, but **improves ranking**: PR-AUC 0.79→0.93 and recall at a 2% false-alarm budget **8%→46%** — at any fixed low false-alarm rate the fixed model catches far more. A simple linear model with the right data out-ranks a fine-tuned transformer with the wrong data.

**2. The model learns the source, not the label.** A TF-IDF probe predicts a skill's *source benchmark* from its text at **91.7% accuracy** (76% balanced, chance 10%). Tokens like `openclaw` act as "benign" cues only because of how the benchmark's sources were mixed.

**3. The fix: train only on sources that contain both labels.** A **size-matched random control stays at 58%**, so it is *which* data, not *less* data. Ablations (with controls) show it is **label-mixing, not minimal-pair construction**, and locate **MalSkillBench** as the single-label source that reintroduces the shortcut; masking the `openclaw` token or stripping YAML frontmatter does not fix it (the shortcut is distributed).

**4. The shortcut is an attack surface.** Appending **8** of the detector's own benign source-style tokens to a malicious skill evades it **17%** of the time vs **5%** for random tokens (a good-word attack, Lowd & Meek 2005); the fix lowers this to 7%. One bias → over-blocking benign *and* evadable malicious → one data fix moves both.

**5. It holds on real data.** On **400 public GitHub `SKILL.md` files** (32% of which contain the `openclaw` token), the merged-data detector flags **30%** as malicious; the fix flags **5%**.

## Reproduce
```bash
uv sync
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ProtectSkills/MaliciousSkillBench', repo_type='dataset', local_dir='data/raw/msb')"
uv run python scripts/baselines.py        # baselines across the three splits
uv run python scripts/shortcut_probe.py   # source is predictable from text
uv run python scripts/robust_classical.py # the mixed-only fix + controls
./scripts/run_transformer_suite.sh        # ModernBERT fine-tune (both training sets)
uv run python scripts/mechanism.py        # pairing / dose-response / masking ablations
uv run python scripts/compare_models.py   # all models, paired bootstrap CIs
uv run python scripts/adv_evasion.py      # good-word evasion attack (dose-response)
uv run python scripts/github_audit.py     # real-world audit on public GitHub skills
uv run python scripts/train_lora.py --help   # LoRA/QLoRA fine-tune (GPU; see docs/RUN_ON_GPU.md)
```

## Serve
```bash
uv run python scripts/export_onnx.py --run modernbert_source_disjoint_mixed_only_s0  # -> models/serve/ (int8)
docker build -t skillshift . && docker run -p 8000:8000 skillshift                   # FastAPI + ONNX Runtime
curl http://localhost:8000/healthz
```
