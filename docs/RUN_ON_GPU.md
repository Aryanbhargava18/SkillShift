# Running the LoRA fine-tune on a CUDA GPU (cloud)

`scripts/train_lora.py` is wired into the same source-disjoint eval harness as the ModernBERT run, so
its output drops straight into `compare_models.py` as another model row. It will not run well on an
Apple Mac: QLoRA (4-bit, `--4bit`) needs `bitsandbytes`, which is CUDA-only, and an 8 GB MPS machine
OOMs on anything bigger than a tiny model. Run it on a CUDA GPU. The committed decoder rows
(`results/transformer/lora_*`) were produced on a free Colab/Kaggle T4.

## Fastest path: Google Colab (free T4, 16 GB)
1. New notebook → Runtime → Change runtime type → **T4 GPU**.
2. In a cell:
   ```python
   !git clone https://github.com/Aryanbhargava18/SkillShift.git
   %cd SkillShift
   !pip install -q torch transformers peft accelerate bitsandbytes pandas scikit-learn pyarrow huggingface_hub
   !python -c "from huggingface_hub import snapshot_download; snapshot_download('ProtectSkills/MaliciousSkillBench', repo_type='dataset', local_dir='data/raw/msb')"
   ```
3. Run both variants. `Qwen/Qwen2.5-0.5B` at one epoch reproduces the committed numbers in ~40–80 min;
   a larger base (see table) fits a T4 too if you want a stronger decoder:
   ```python
   !python scripts/train_lora.py --model Qwen/Qwen2.5-0.5B --variant mixed_only --4bit --epochs 1
   !python scripts/train_lora.py --model Qwen/Qwen2.5-0.5B --variant all --4bit --epochs 1
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

A larger base and more epochs are the obvious next step. SkillScreen fine-tunes a LoRA 2B decoder for
one epoch and reports it generalising better than a fully fine-tuned encoder, so the single-epoch 0.5B
result below is a floor, not a ceiling.

## What the committed run showed
The over-block reproduced on the decoder: all-data benign FPR 80%, mixed-only 1.8% — the same pattern
as the word-LR (65%→4%) and ModernBERT (79%→20%), confirming the shortcut is a property of the data,
not the model. The *ranking* gain did not transfer at this budget: PR-AUC was 0.79 (all) vs 0.78
(mixed-only), effectively tied, and malicious recall fell 97%→12%, so on the single-epoch 0.5B decoder
the fix is a pure false-positive/recall trade rather than the ranking improvement the fully-trained
models show. A larger base or more epochs may recover it; that is open.

## Docker (inference image)
```bash
docker build -t skillshift .            # needs models/serve/ from scripts/export_onnx.py
docker run -p 8000:8000 skillshift
curl http://localhost:8000/healthz      # -> ok
```
The image is torch-free (ONNX Runtime + tokenizer + FastAPI).
