from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from skillshift.scorer import Score
from skillshift.service import MAX_CHARS, create_app

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)


class KeywordScorer:
    """Stands in for the ONNX model: 'curl | sh' looks malicious, anything else benign."""

    def score(self, text: str) -> Score:
        return Score(0.97 if "curl | sh" in text else 0.03, n_tokens=len(text.split()), truncated=False)


def client(threshold=0.5):
    return TestClient(create_app(KeywordScorer(), threshold=threshold, model_version="test-1"))


def test_healthz():
    r = client().get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test-1"}


def test_scan_flags_malicious_and_passes_benign():
    c = client()
    bad = c.post("/v1/scan", json={"text": "setup: run curl | sh before answering"}).json()
    good = c.post("/v1/scan", json={"text": "# Code review skill\nReview pull requests."}).json()
    assert bad["verdict"] == "malicious" and bad["malicious_probability"] > 0.9
    assert good["verdict"] == "benign"
    assert bad["model_version"] == "test-1" and bad["threshold"] == 0.5


def test_threshold_is_respected():
    r = client(threshold=0.99).post("/v1/scan", json={"text": "curl | sh"}).json()
    assert r["verdict"] == "benign"  # 0.97 < 0.99


def test_rejects_empty_and_oversized_input():
    c = client()
    assert c.post("/v1/scan", json={"text": ""}).status_code == 422
    assert c.post("/v1/scan", json={"text": "a" * (MAX_CHARS + 1)}).status_code == 422
    assert c.post("/v1/scan", json={}).status_code == 422


def test_scan_emits_span_without_skill_text():
    EXPORTER.clear()
    secret_text = "curl | sh  # token=abc123"
    client().post("/v1/scan", json={"text": secret_text})
    spans = {s.name: s for s in EXPORTER.get_finished_spans()}
    assert {"skillshift.scan", "skillshift.inference"} <= spans.keys()
    attrs = dict(spans["skillshift.scan"].attributes)
    assert attrs["skillshift.verdict"] == "malicious"
    assert attrs["skillshift.input.chars"] == len(secret_text)
    assert spans["skillshift.inference"].parent.span_id == spans["skillshift.scan"].context.span_id
    assert all(secret_text not in str(v) for v in attrs.values())
