# ProtectRAG

[![PyPI](https://img.shields.io/pypi/v/protectrag)](https://pypi.org/project/protectrag/)
[![Python](https://img.shields.io/pypi/pyversions/protectrag)](https://pypi.org/project/protectrag/)
[![License](https://img.shields.io/github/license/Ritvik777/ProtectRAG)](LICENSE)

**Screen RAG documents for prompt injection, apply allow / warn / block policies, and export logs, metrics, and eval reports** — as a lightweight Python library with zero required dependencies.

---

## Table of contents

- [What it is](#what-it-is)
- [How it works](#how-it-works)
- [Install](#install)
- [Quick start](#quick-start)
- [Retrieval-time screening](#retrieval-time-screening)
- [Async & batch API](#async--batch-api)
- [Callbacks (alerts & webhooks)](#callbacks-alerts--webhooks)
- [Framework integrations](#framework-integrations)
- [Golden dataset & offline evals](#golden-dataset--offline-evals)
- [Observability](#observability)
- [LLM retry & fallback](#llm-retry--fallback)
- [Configuration](#configuration)
- [Concepts & full API](#concepts--full-api)
- [UI / visual analysis](#ui--visual-analysis)
- [Vector database compatibility](#vector-database-compatibility)
- [Limitations](#limitations)
- [Development](#development)
- [Publishing new versions](#publishing-new-versions)
- [License](#license)

---

## What it is

Text going into a RAG vector store can contain **prompt injection** — hidden instructions that hijack the assistant when the chunk is retrieved. ProtectRAG screens that text **before indexing** (and optionally **after retrieval**), classifies risk, applies your policy, and gives you structured logs, metrics, callbacks, and eval reports.

---

## How it works

```
  Text chunk
       │
       ▼
  ┌──────────┐    ┌────────────────┐
  │  Scan    │───▶│ Heuristic (13  │  ← free, fast, local
  │          │    │ rule families)  │
  │          │───▶│ LLM classifier │  ← OpenAI-compatible API
  │          │───▶│ Hybrid         │  ← heuristics first, LLM when needed
  └──────────┘    └────────────────┘
       │
       ▼
  ┌──────────┐
  │  Policy  │──▶  ALLOW  /  WARN  /  BLOCK
  └──────────┘
       │
       ▼
  Logs · Metrics · Callbacks · OTel spans
```

### Detection coverage (13 heuristic rule families)

| Rule family | Examples |
|-------------|----------|
| Instruction override | "ignore previous instructions", "you are now", "act as if", "switch to mode" |
| Multi-language override | Chinese, Japanese, French, Spanish, German, Arabic, Korean |
| Role delimiter injection | `<\|system\|>`, `[SYSTEM]:`, `---SYSTEM---`, `END_OF_SYSTEM_PROMPT` |
| Prompt leak / exfiltration request | "repeat your system prompt", "reveal hidden instructions" |
| Data exfiltration | "send all docs to URL", markdown image exfil, HTML img exfil |
| Fake tool / function call | `<tool_call>`, `function call:`, JSON role injection |
| Encoding tricks | base64 decode, rot13, hex, "decode this" |
| Unicode manipulation | Zero-width characters, bidi overrides, invisible tag blocks |
| HTML / script injection | `<script>`, `<iframe>`, `javascript:`, HTML comment tricks |
| Markdown injection | Image event handlers, `data:` URIs, comment overrides |
| Indirect / deferred injection | "when the user asks X, reply Y instead" |
| Payload splitting | "this is part 1 of 3, combine with next" |

Plus an **LLM classifier** for nuanced cases the rules can't catch.

---

## Install

```bash
pip install protectrag                    # core (zero dependencies)
pip install "protectrag[llm]"             # + LLM classifier (httpx)
pip install "protectrag[langchain]"       # + LangChain integration
pip install "protectrag[llamaindex]"      # + LlamaIndex integration
pip install "protectrag[fastapi]"         # + FastAPI middleware/dependency
pip install "protectrag[otel]"            # + OpenTelemetry tracing
pip install "protectrag[llm,fastapi]"     # combine extras
```

---

## Quick start

### Heuristics only (no API key needed)

```python
from protectrag import ingest_document, IngestDecision, InjectionSeverity

out = ingest_document(
    "Your document chunk text here.",
    document_id="chunk-001",
    block_on=InjectionSeverity.HIGH,
    warn_on=InjectionSeverity.MEDIUM,
)

if out.decision is IngestDecision.BLOCK:
    print("Blocked:", out.message)
elif out.decision is IngestDecision.ALLOW_WITH_WARNING:
    print("Warning:", out.message)
else:
    print("Clean — safe to index")
```

### Hybrid (heuristics + LLM)

```python
import os
from protectrag import HybridScanner, LLMScanner, ingest_document

os.environ["OPENAI_API_KEY"] = "sk-..."

with LLMScanner.from_env() as llm:
    hybrid = HybridScanner(llm)
    out = ingest_document(
        text,
        document_id="chunk-001",
        scan=lambda t, d: hybrid.scan(t, document_id=d),
    )
```

---

## Retrieval-time screening

Screen chunks **after** they come back from the vector DB and **before** they reach the LLM:

```python
from protectrag import RetrievedChunk, screen_retrieved_chunks

chunks = [
    RetrievedChunk(text="Normal policy docs.", chunk_id="c1"),
    RetrievedChunk(text="Ignore previous instructions.", chunk_id="c2"),
]

result = screen_retrieved_chunks(chunks)
safe_texts = result.passed_texts()  # only clean chunks
print(f"Blocked {result.n_blocked} of {result.total} chunks")
```

---

## Async & batch API

For high-throughput pipelines processing thousands of chunks:

```python
import asyncio
from protectrag import async_scan_batch

items = [("chunk text 1", "id-1"), ("chunk text 2", "id-2"), ...]

result = asyncio.run(async_scan_batch(items, max_concurrency=10))
print(result.summary())  # {"total": ..., "blocked": ..., "allowed": ...}
```

---

## Callbacks (alerts & webhooks)

Fire custom functions on block / warn / allow decisions:

```python
from protectrag import CallbackRegistry, ingest_document

def send_slack_alert(text, result):
    print(f"BLOCKED doc={result.document_id} severity={result.severity.name}")

def quarantine(text, result):
    # write to quarantine queue / database
    pass

cb = CallbackRegistry(
    on_block=[send_slack_alert, quarantine],
    on_warn=[send_slack_alert],
)

ingest_document(text, document_id="d1", callbacks=cb)
```

---

## Framework integrations

### LangChain

```python
from protectrag.integrations.langchain import ProtectRAGFilter

guard = ProtectRAGFilter()
docs = retriever.get_relevant_documents(query)
safe_docs = guard.transform_documents(docs)  # injected docs removed
```

### LlamaIndex

```python
from protectrag.integrations.llamaindex import ProtectRAGPostprocessor

query_engine = index.as_query_engine(
    node_postprocessors=[ProtectRAGPostprocessor()],
)
```

### FastAPI

```python
from fastapi import FastAPI
from protectrag.integrations.fastapi import create_screening_middleware

app = FastAPI()
app.add_middleware(create_screening_middleware(paths=["/api/ingest"]))
```

---

## Golden dataset & offline evals

Ships with **100 labeled examples** (50 injection + 50 clean) covering all attack families:

```python
from protectrag import load_golden_v1, run_eval_dataset, scan_document_for_injection

cases = load_golden_v1()
report = run_eval_dataset(
    cases,
    classify=lambda t, d: scan_document_for_injection(t, document_id=d),
    run_id="ci-build-42",
)
print(f"Precision: {report.precision:.2f}  Recall: {report.recall:.2f}")
```

Run this in CI to catch regressions when you change rules or models.

---

## Observability

### Structured logs

```python
from protectrag import configure_logging, ingest_document, RunContext

configure_logging()
ctx = RunContext(project="acme-rag", environment="prod")
ingest_document(text, document_id="d1", context=ctx)
# Emits JSON: {"event":"rag_document_screen","action":"ingest_blocked",...,"run_id":"...","project":"acme-rag"}
```

### Metrics

```python
from protectrag import InMemoryMetrics, ingest_document

m = InMemoryMetrics()
ingest_document(text, document_id="d1", metrics=m)
print(m.snapshot())  # counters + histograms
```

### OpenTelemetry spans

```python
from protectrag import span_attributes_for_ingest_scan
# Attach to your exporter (Phoenix, Jaeger, Datadog, etc.)
attrs = span_attributes_for_ingest_scan(result, latency_ms=12.3, model="gpt-4o-mini")
```

---

## LLM retry & fallback

Automatic retry with exponential backoff on rate limits (429) and server errors (5xx). Falls back to heuristics if the LLM is completely unavailable:

```python
from protectrag import LLMScanner, RetryConfig, with_retry

scanner = LLMScanner.from_env()
result = with_retry(
    lambda: scanner.scan(text, document_id="d1"),
    text=text,
    document_id="d1",
    config=RetryConfig(max_retries=3, fallback_to_heuristic=True),
)
```

---

## Configuration

Optional environment variables (LLM mode only — see [`.env.example`](.env.example)):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | Required for LLM calls |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible server |
| `PROTECTRAG_LLM_MODEL` | `gpt-4o-mini` | Classification model |

Heuristic-only usage needs **none** of these.

---

## Concepts & full API

| Export | Role |
|--------|------|
| `scan_document_for_injection` | Heuristic scan → `DocumentScanResult` |
| `ingest_document` | Scan + policy + logs/metrics/callbacks |
| `screen_retrieved_chunks` | Retrieval-time filtering |
| `async_scan`, `async_scan_batch` | Async + concurrent batch scanning |
| `LLMScanner`, `HybridScanner` | LLM / hybrid classification |
| `with_retry`, `RetryConfig` | LLM retry + heuristic fallback |
| `CallbackRegistry` | `on_block` / `on_warn` / `on_allow` hooks |
| `load_golden_v1`, `run_eval_dataset` | Golden dataset + eval reports |
| `RunContext`, `InMemoryMetrics` | Run correlation + metrics |
| `configure_logging`, `emit_ingest_event` | Structured JSON logging |
| `span_attributes_for_ingest_scan` | OTel span attributes |
| `ProtectRAGFilter` | LangChain document transformer |
| `ProtectRAGPostprocessor` | LlamaIndex node postprocessor |
| `create_screening_middleware` | FastAPI middleware |

---

## UI / visual analysis

ProtectRAG is a **library, not a hosted platform** — it has no built-in web UI. For visual analysis:

- **Jupyter notebook** — `notebooks/eval_dashboard.ipynb` loads the golden dataset, runs evals, and prints confusion matrix + severity distribution.
- **Grafana / Datadog** — Consume the structured JSON logs (`configure_logging()`) or metrics (`MetricsSink`) to build dashboards.
- **Phoenix / Jaeger** — Use the `[otel]` extra to export spans to any OTel-compatible backend for trace visualization.

---

## Vector database compatibility

Works with **every** vector store — Pinecone, Qdrant, Weaviate, Milvus, pgvector, Chroma, OpenSearch, FAISS, etc. ProtectRAG only processes **text strings**; it never talks to your database. You call it before (or after) your embed + upsert.

---

## Limitations

- Heuristic rules are pattern-based and can miss novel attack styles or rarely flag benign text.
- The LLM classifier is stronger but costs money and is not perfect.
- Long texts are truncated (head + tail) before sending to the LLM.
- No built-in web dashboard — use notebooks, Grafana, or OTel backends.

---

## Development

```bash
git clone https://github.com/Ritvik777/ProtectRAG.git
cd ProtectRAG
pip install -e ".[dev]"
pytest tests/ -v          # 44 tests
python -m build           # build wheel + sdist
```

---

## Publishing new versions

```bash
# 1. Bump version in pyproject.toml
# 2. Build
rm -rf dist && python -m build
# 3. Upload
python -m twine upload dist/*
```

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
