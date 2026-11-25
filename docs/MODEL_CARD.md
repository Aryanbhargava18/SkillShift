# Model card — SkillShift reference scanner

**Model.** ModernBERT-base fine-tuned on MaliciousSkillBench, `source_disjoint` split, **mixed-label sources only** (seed 0), exported to int8 ONNX (`scripts/export_onnx.py`). This is the `modernbert_source_disjoint_mixed_only_s0` run. It is not the lowest-FPR detector in the paper — the word-LR reaches 3.7% benign FPR — but it is the one shipped as a scanner.

**Input / output.** Input is `SKILL.md` text (first 512 tokens). Output is a malicious probability in [0, 1].

**Operating threshold.** 0.344, calibrated to a 2% benign false-positive rate on validation data. The default 0.5 threshold is also available. Reported behaviour at 0.5 on the unseen-source test set: benign FPR 4% on ATR and 20% aggregate; malicious recall ~53%.

**Intended use.** A cheap static pre-install triage filter that routes suspicious skills to human review. It is a first-pass signal, not a standalone verdict.

**Key limitation — thresholds do not transfer.** Validation data comes from the training sources, so a threshold calibrated there under-states the false-positive rate on sources the model has not seen. On a new benign source the realized FPR can be higher than 20%; recalibrate on in-domain benign skills before relying on a fixed threshold. The same mixed-only data recipe, measured with a linear detector on 400 unlabeled public GitHub skills, cuts the flag rate 30%→5%; the ModernBERT scanner itself has not been audited on that set.

**Out of scope.** It reads the skill text only. Payloads hidden from static text — bundled code, obfuscation, instructions that only execute at runtime — need dynamic or package-level analysis. The good-word attack in the paper shows that appending benign source-style tokens can push a caught malicious skill below threshold, so this filter should not be the only line of defense.

**Training data.** MaliciousSkillBench (CC-BY-4.0, arXiv:2608.19901); benign training skills come almost entirely from one source (SkillTrustBench), so the fix is in practice "train on SkillTrustBench."
