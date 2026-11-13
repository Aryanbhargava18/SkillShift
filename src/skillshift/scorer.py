"""Score a skill with the exported ONNX model. Needs only onnxruntime + tokenizers (no torch).

A model directory (made by scripts/export_onnx.py) contains:
    model.onnx        the classifier (int8-quantised by default)
    tokenizer.json    the ModernBERT tokenizer
    skillshift.json   {"model_version", "threshold", "max_length"}
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


@dataclass
class Score:
    probability: float  # P(malicious)
    n_tokens: int  # tokens fed to the model, after truncation
    truncated: bool  # True if the skill was longer than max_length and its tail was not read


class OnnxScorer:
    def __init__(self, model_dir: str | Path, model_file: str = "model.onnx", threads: int = 0):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "skillshift.json").read_text())
        self.max_length = int(self.config["max_length"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(self.max_length)
        self.tokenizer.no_padding()
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_dir / model_file), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def score(self, text: str) -> Score:
        enc = self.tokenizer.encode(text)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        logits = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0][0]
        exp = np.exp(logits - logits.max())
        return Score(float(exp[1] / exp.sum()), len(enc.ids), bool(enc.overflowing))
