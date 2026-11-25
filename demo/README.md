---
title: SkillShift
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

Interactive demo for [SkillShift](https://github.com/Aryanbhargava18/SkillShift): paste a `SKILL.md`
and watch a detector trained on the whole merged benchmark (which keys on source style) disagree with
one trained only on both-label sources (the fix).

**Deploy to Hugging Face Spaces:**
```bash
pip install -U huggingface_hub
huggingface-cli login            # your HF token
huggingface-cli repo create skillshift --type space --space_sdk gradio
cd demo && git init && git add . && git commit -m "SkillShift demo"
git remote add hf https://huggingface.co/spaces/<your-username>/skillshift
git push hf HEAD:main
```
