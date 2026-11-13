"""HTTP service: POST a SKILL.md, get back whether it looks malicious.

Run locally:
    SKILLSHIFT_MODEL_DIR=models/serve uv run uvicorn skillshift.service:app --app-dir src

Tracing: set OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://localhost:4318) to send OpenTelemetry
spans to a collector. Each scan creates a "skillshift.scan" span with child spans for
tokenisation+inference, carrying the input size, truncation, score and verdict as attributes.
The skill text itself is never put on a span.
"""

import os
import time
from typing import Literal, Protocol

from fastapi import FastAPI
from opentelemetry import trace
from pydantic import BaseModel, Field

from skillshift.scorer import Score

MAX_CHARS = 500_000


class Scorer(Protocol):
    def score(self, text: str) -> Score: ...


class ScanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS, description="Full SKILL.md content")


class ScanResponse(BaseModel):
    verdict: Literal["malicious", "benign"]
    malicious_probability: float
    threshold: float
    truncated: bool
    n_tokens: int
    model_version: str
    latency_ms: float


def create_app(scorer: Scorer, threshold: float, model_version: str) -> FastAPI:
    app = FastAPI(title="skillshift", version="0.1.0")
    tracer = trace.get_tracer("skillshift")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "model_version": model_version}

    @app.post("/v1/scan", response_model=ScanResponse)
    def scan(req: ScanRequest) -> ScanResponse:
        start = time.perf_counter()
        with tracer.start_as_current_span("skillshift.scan") as span:
            span.set_attribute("skillshift.model.version", model_version)
            span.set_attribute("skillshift.input.chars", len(req.text))
            with tracer.start_as_current_span("skillshift.inference"):
                result = scorer.score(req.text)
            verdict = "malicious" if result.probability >= threshold else "benign"
            span.set_attribute("skillshift.input.tokens", result.n_tokens)
            span.set_attribute("skillshift.input.truncated", result.truncated)
            span.set_attribute("skillshift.score", result.probability)
            span.set_attribute("skillshift.verdict", verdict)
        return ScanResponse(
            verdict=verdict,
            malicious_probability=round(result.probability, 6),
            threshold=threshold,
            truncated=result.truncated,
            n_tokens=result.n_tokens,
            model_version=model_version,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    return app


def configure_tracing() -> None:
    """Export spans over OTLP/HTTP when OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise tracing is a no-op."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "skillshift"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def build_from_env() -> FastAPI:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    from skillshift.scorer import OnnxScorer

    configure_tracing()
    model_dir = os.environ.get("SKILLSHIFT_MODEL_DIR", "models/serve")
    scorer = OnnxScorer(model_dir, model_file=os.environ.get("SKILLSHIFT_MODEL_FILE", "model.onnx"))
    app = create_app(scorer, float(scorer.config["threshold"]), scorer.config["model_version"])
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz")
    return app


def __getattr__(name: str):
    # Lets `uvicorn skillshift.service:app` build the app lazily, while tests import create_app
    # without needing a model on disk.
    if name == "app":
        return build_from_env()
    raise AttributeError(name)
