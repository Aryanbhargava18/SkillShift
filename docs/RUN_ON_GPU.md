# Running the LoRA fine-tune on a CUDA GPU (cloud)

`scripts/train_lora.py` is written and wired into the same source-disjoint eval harness as the
ModernBERT run, so its output drops straight into `compare_models.py` as another model row. It will
not run well on an Apple Mac: QLoRA (4-bit, `--4bit`) needs `bitsandbytes`, which is CUDA-only, and
the 8 GB MPS machine OOMs on anything bigger than a tiny model. Run it on a CUDA GPU.

## Fastest path: Google Colab (free T4, 16 GB)
1. New notebook → Runtime → Change runtime type → **T4 GPU**.
2. In a cell:
   ```python
   !git clone https://github.com/Aryanbhargava18/SkillShift.git && cd SkillShift
   %cd SkillShift
   !pip install -q torch transformers peft accelerate bitsandbytes pandas scikit-learn pyarrow huggingface_hub
   !python -c "from huggingface_hub import snapshot_download; snapshot_download('ProtectSkills/MaliciousSkillBench', repo_type='dataset', local_dir='data/raw/msb')"
   ```
3. Run both variants (QLoRA 4-bit so a bigger base fits a T4):
   ```python
   !python scripts/train_lora.py --model Qwen/Qwen2.5-1.5B --variant mixed_only --4bit
   !python scripts/train_lora.py --model Qwen/Qwen2.5-1.5B --variant all --4bit
   ```
4. Fold the LoRA rows into the comparison table (needs the classical scores; run mechanism first if
   `results/classical_test_scores.csv` is absent):
   ```python
   !python scripts/mechanism.py          # writes classical_test_scores.csv (CPU, a few min)
   !python scripts/compare_models.py     # now includes lora_* rows, with bootstrap CIs
   !python scripts/make_paper_numbers.py # regenerates paper/numbers.tex
   ```
5. Download `results/transformer/lora_*` and commit them; the paper picks them up automatically.

## Which base model fits which GPU (QLoRA 4-bit)
| GPU | VRAM | comfortable base model |
|---|---|---|
| T4 (free Colab) | 16 GB | Qwen2.5-0.5B / 1.5B / 3B |
| L4 / A10 | 24 GB | up to 7B |
| A100 | 40–80 GB | 7B–14B, or plain fp16 LoRA (drop `--4bit`) |

Start with **Qwen2.5-1.5B** — it matches the LoRA-Qwen that SkillScreen used on this dataset, so the
comparison is apples-to-apples, and it trains in well under an hour on a T4.

## What to expect
A third model class (LoRA on a decoder LLM) alongside TF-IDF and ModernBERT. If the all-data LoRA also
over-blocks unseen-source benign skills and the mixed-only LoRA does not, it is one more confirmation
that the shortcut is a property of the data, not the model. (The ModernBERT result already shows this:
79% -> 20% benign FPR.) This run also stands alone as the "parameter-efficient fine-tuning + evaluation
harness" artifact that LLM job postings ask for.

## Docker (inference image) — verified working locally
```bash
docker build -t skillshift .            # needs models/serve/ from scripts/export_onnx.py
docker run -p 8000:8000 skillshift
curl http://localhost:8000/healthz      # -> ok
```
The image is torch-free (ONNX Runtime + tokenizer + FastAPI). Built and healthchecked green on
2026-11-24.
