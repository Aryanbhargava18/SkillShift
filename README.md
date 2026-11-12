# SkillShift

Can a fine-tuned model detect malicious agent skills (SKILL.md files) from sources it has never seen?

Work in progress. Working notes and results so far are in [NOTES.md](NOTES.md).

## Data

[MaliciousSkillBench](https://huggingface.co/datasets/ProtectSkills/MaliciousSkillBench) (CC-BY-4.0).
It is not stored in this repo. Download it with:

```bash
uv sync
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('ProtectSkills/MaliciousSkillBench', repo_type='dataset', local_dir='data/raw/msb')"
```

## Run

```bash
uv run python scripts/inspect_data.py   # step 1: dataset summary and split checks
```
