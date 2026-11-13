"""Exercise OnnxScorer end to end with a tiny hand-built model, so CI needs no real weights.

The toy model's malicious logit is mean(token ids) - 2, so the word "evil" (id 5)
pushes towards malicious and "safe" (id 1) towards benign.
"""

import json

import numpy as np
import onnx
from onnx import TensorProto, helper
from tokenizers import Tokenizer, models, pre_tokenizers

from skillshift.scorer import OnnxScorer


def build_model_dir(tmp_path, max_length=8):
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "safe": 1, "evil": 5}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "tokenizer.json"))

    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["ids_f"], ["mean"], keepdims=1),  # [1, 1]
        helper.make_node("Sub", ["mean", "two"], ["mal"]),
        helper.make_node("Mul", ["mal", "zero"], ["benign"]),
        helper.make_node("Concat", ["benign", "mal"], ["logits"], axis=1),  # [1, 2]
    ]
    graph = helper.make_graph(
        nodes, "toy",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, None]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, None])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])],
        initializer=[helper.make_tensor("two", TensorProto.FLOAT, [], [2.0]),
                     helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, tmp_path / "model.onnx")
    (tmp_path / "skillshift.json").write_text(
        json.dumps({"model_version": "toy", "threshold": 0.5, "max_length": max_length}))
    return tmp_path


def test_scores_follow_model_output(tmp_path):
    scorer = OnnxScorer(build_model_dir(tmp_path))
    evil, safe = scorer.score("evil evil"), scorer.score("safe safe")
    assert np.isclose(evil.probability, 1 / (1 + np.exp(-3)))  # logit 5 - 2 = 3
    assert np.isclose(safe.probability, 1 / (1 + np.exp(1)))  # logit 1 - 2 = -1
    assert evil.n_tokens == 2 and not evil.truncated


def test_long_input_is_truncated_and_reported(tmp_path):
    scorer = OnnxScorer(build_model_dir(tmp_path, max_length=3))
    s = scorer.score("safe safe safe evil evil evil")
    assert s.n_tokens == 3 and s.truncated
    assert np.isclose(s.probability, 1 / (1 + np.exp(1)))  # only "safe safe safe" was read
