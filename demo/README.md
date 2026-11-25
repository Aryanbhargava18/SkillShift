# SkillShift demo

Two detectors for malicious agent skills — one trained on the whole merged benchmark (which keys on source
*style* and over-blocks) and one trained only on both-label sources (the fix) — side by side. Paste a
`SKILL.md` and watch them disagree.

**Live:** https://huggingface.co/spaces/Aryanwgaf/SkillShift

## Two equivalent builds

Both serve the *same* fitted word-TF-IDF + logistic-regression models; they differ only in where the math runs.

- **`web/` — static, client-side (this is what's deployed).** `web/index.html` loads `web/models.json` (the
  fitted vocab, idf, and coefficients exported by `scripts/export_static_models.py`) and reimplements
  `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, norm="l2")` + logistic regression in JavaScript. It
  matches scikit-learn's `predict_proba` to < 5e-7. No server, no cold start — which is why it, not the Gradio
  app, is on the free Hugging Face Space (free accounts can only host *static* Spaces).
- **`app.py` — Gradio, server-side (for local use).** `python app.py` loads the precomputed models from
  `models.joblib` and serves the same comparison via Gradio. Regenerate the pickle with
  `python scripts/build_demo_models.py`.

## Regenerate the exported models
```bash
python scripts/build_demo_models.py      # -> demo/models.joblib (for app.py)
python scripts/export_static_models.py    # -> demo/web/models.json
```
