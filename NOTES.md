# skillshift: working notes

Question: can a fine-tuned model detect malicious agent skills from sources it has never seen?

## Step 1: the data (2026-11-12)

Dataset: MaliciousSkillBench (Hugging Face `ProtectSkills/MaliciousSkillBench`, CC-BY-4.0, 35 MB).
We use `primary.parquet` (9,740 skills) and the frozen split files in `splits/`.

What it contains:
- 9,740 skills: 7,505 malicious (label 1), 2,235 benign (label 0). Real life is the other way round
  (Snyk found about 13% risky on registries), so accuracy here says little about real-world use.
- `skill_text`: the SKILL.md text. Median about 5,000 characters (roughly 1,250 tokens), p90 about 13,700.
- 13 source benchmarks merged together. Most sources are all-malicious (MalSkillBench 3,426, SkillHarm 728)
  or nearly all-benign (ATR 455 benign / 21 malicious).

Why that matters (the core problem): a model can learn "this looks like a SkillHarm file" instead of
"this is malicious". That is a shortcut, not detection.

The three evaluation protocols:
- `random`: train and test drawn from the same mix. Easy and optimistic.
- `m_structural_disjoint`: near-copies (same structural family) never in both train and test. Stops memorising copies.
- `source_disjoint`: test sources never seen in training. Test = SkillHarm (728 malicious), ATR (455 benign,
  21 malicious), SkillFortifyBench (90 / 90). Almost all training benign skills come from one source
  (SkillTrustBench, 1,478), so the model must recognise benign skills written in styles it has never seen.

Published baseline (paper, arXiv 2608.19901): word TF-IDF + SVM gets macro-F1 0.932 on random but 0.665 on
source-disjoint, with 62.4% of unseen-source benign skills wrongly flagged.

Our goal: close that gap. The main number we care about is source-disjoint performance, especially the
false-positive rate on benign skills from unseen sources.

## Check yourself before step 2
1. Why is "97% accuracy on the random split" not impressive here?
2. What is a shortcut, and which column in the data would let you test for it?
3. Why does the source-disjoint split mostly test benign skills, not malicious ones?

## Step 2: baselines (2026-11-13)

`uv run python scripts/baselines.py` writes `results/baselines.json`. Macro-F1 / share of benign skills wrongly flagged:

| model | random | structural-disjoint | source-disjoint |
|---|---|---|---|
| keyword rules | 0.456 / 37% | 0.465 / 39% | 0.297 / 44% |
| TF-IDF + logistic regression | 0.876 / 11% | 0.849 / 13% | 0.653 / 64% |
| TF-IDF + linear SVM (paper's baseline) | 0.932 / 10% | 0.907 / 11% | 0.647 / 66% |

We reproduce the paper: the SVM gets 0.932 on random (paper 0.932) and 0.647 on source-disjoint (paper 0.665),
flagging 66% of unseen-source benign skills (paper 62.4%). So our evaluation setup matches theirs.
At 1% false positives, source-disjoint recall falls to about 4-6%: these models can't be deployed on new sources.

## Step 3: proving the shortcut exists (2026-11-14)

`uv run python scripts/shortcut_probe.py` writes `results/shortcut_probe.json`.

1. **Can the text reveal the source?** A TF-IDF + logistic regression model trained to name the source
   benchmark gets 91.7% accuracy (76.0% balanced across 10 sources; chance is 10%). So source style is
   easy to learn.
2. **Does the malicious/benign SVM use it?** Its strongest "benign" words include `openclaw`,
   `metadata openclaw` and `openclaw workspace`: boilerplate from skills published for the OpenClaw agent.
3. **Why would that word look benign?** Share of training skills containing `openclaw`:

   | training data | benign | malicious |
   |---|---|---|
   | all training sources | 44.9% | 22.3% |
   | inside SkillTrustBench only | 46.2% | 45.1% |
   | mixed-label sources only | 44.9% | 44.0% |

   Inside SkillTrustBench, the word says nothing: it's equally common in both labels. It only looks
   "benign" because benign training data is mostly SkillTrustBench, while the biggest malicious source
   (MalSkillBench) rarely mentions OpenClaw (5%). The correlation comes from **how the sources were mixed**,
   not from what makes a skill dangerous. ATR's benign skills (the unseen test source) mostly don't
   mention OpenClaw (9%), so the model loses this "safe" cue. `openclaw` is the clearest example, not the
   only one: the model uses many such style cues together, and their combined effect is the 66% false-positive rate.

**Check yourself:** Why is `openclaw` a shortcut feature, not a real signal? Which row of the table
proves it? What would happen if an attacker added OpenClaw boilerplate to a malicious skill?

## Step 4: fixing the data, not the model (2026-11-14)

`uv run python scripts/robust_classical.py` writes `results/robust_classical.json`.
All numbers are on source-disjoint test (sources never seen in training).

Idea: if a source contains only one label, it can only teach style. If a source contains both
labels, style can't separate them inside it, so the model has to learn content. So we tried:

| training data | benign wrongly flagged | PR-AUC |
|---|---|---|
| all 7,513 training skills (paper's setting), word LR | 65.0% | 0.793 |
| re-weighted so each (source, label) group counts equally | 62-79% | 0.83-0.89 |
| **only mixed-label sources** (4,092 skills), word LR | **3.7%** | **0.928** |
| control: random 4,092 skills, same label ratio (3 seeds), word LR | 58.0% | 0.800 |

- Re-weighting doesn't help: the style signal is still there, just weighted differently.
- Training on *less* data, but the *right* data, cuts false alarms on unseen-source benign skills ~17x.
- The control proves it isn't "less data". Random subsets of the same size still flag 58%.
- Cost: at the default threshold it catches fewer malicious skills (50% vs 96%). But its ranking is far
  better (PR-AUC 0.93 vs 0.79), so at any fixed false-alarm budget it catches more:
  at 2% FPR, 46% vs 8%.

**Calibration does not transfer.** Picking the threshold that flags 2% of *validation* benign skills
(validation comes from training sources) gives 22% on unseen test sources for the all-data SVM.
Any number reported as "at 1% FPR" has to say where the threshold came from.

**Check yourself:**
1. Why does the size-matched control matter? What claim would be unsupported without it?
2. Mixed-only catches fewer malicious skills at threshold 0.5 but has a higher PR-AUC. How can both be true?
3. Why can't we trust a threshold tuned on validation here?

## Steps 5+ (written up in the paper)

These notes stop at the classical fix. The later steps — the ModernBERT fine-tune, the Qwen2.5-0.5B
QLoRA decoder, the mechanism ablations (minimal-pair vs label-mixing, the dose-response, token masking),
the good-word evasion attack, the real-GitHub audit, and the ONNX scanner — are in `paper/skillshift.pdf`,
with every number generated from `results/*.json`. One-line summary: the over-block reproduces on all three
model classes (word-LR, encoder, decoder); the data fix cuts unseen-source benign FPR each time; the fix's
ranking gain shows on the two fully-trained models but not on the single-epoch decoder; and the same
shortcut is an evasion channel the fix also narrows.
