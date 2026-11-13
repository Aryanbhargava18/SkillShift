# Inference image: ONNX Runtime + tokenizer + FastAPI. No torch, no training code.
# Build after scripts/export_onnx.py has written models/serve/:
#   docker build -t skillshift .
#   docker run -p 8000:8000 skillshift
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --only-group serve --no-hashes --no-emit-project -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt

COPY src ./src
COPY models/serve ./models/serve

ENV PYTHONPATH=/app/src \
    SKILLSHIFT_MODEL_DIR=/app/models/serve \
    PYTHONUNBUFFERED=1

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "skillshift.service:app", "--host", "0.0.0.0", "--port", "8000"]
