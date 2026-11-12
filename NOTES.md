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
