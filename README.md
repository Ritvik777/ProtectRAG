# ProtectRAG

[![PyPI](https://img.shields.io/pypi/v/protectrag)](https://pypi.org/project/protectrag/)
[![Python](https://img.shields.io/pypi/pyversions/protectrag)](https://pypi.org/project/protectrag/)
[![License](https://img.shields.io/github/license/Ritvik777/ProtectRAG)](LICENSE)

**Screen RAG documents for prompt injection, apply allow / warn / block policies, and export logs, metrics, and eval reports** — as a lightweight Python library with zero required dependencies.

**Quick links:** [Website](https://ritvik777.github.io/ProtectRAG/) (GitHub Pages — enable in repo Settings → Pages → `/docs`) · [PyPI](https://pypi.org/project/protectrag/) · [Issues](https://github.com/Ritvik777/ProtectRAG/issues) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

---

## Table of contents

- [Requirements](#requirements)
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
- [Community & support](#community--support)
- [Contributing](#contributing)
- [Security](#security)
- [Development](#development)
- [Repository layout](#repository-layout)
- [Publishing new versions](#publishing-new-versions)
- [License](#license)

---

## Requirements

- **Python 3.10+** (see [`pyproject.toml`](pyproject.toml))
- **Core install:** no mandatory third-party dependencies
- **Optional extras:** `llm`, `langchain`, `llamaindex`, `fastapi`, `otel`, `redis` — see [Install](#install)

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
pip install "protectrag[redis]"           # + Redis-backed LLM result cache (multi-replica)
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

**Shared Redis cache (multi-replica):** `pip install "protectrag[redis]"`, then pass `RedisLLMClassificationCache(redis.Redis(...))` as `LLMScanner(..., shared_cache=cache)` or `LLMScanner.from_env(shared_cache=cache)` so workers dedupe identical bodies and reduce LLM spend.

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

Use `screen_retrieved_chunks(..., max_workers=8)` for parallel **sync** screening in thread pools, or `await screen_retrieved_chunks_async(..., max_concurrency=10)` (and optional `async_scan_fn=hybrid.ascan`) in async apps.

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

- `async_scan_batch(..., batch_chunk_size=2000)` avoids scheduling millions of tasks at once.
- Pass `async_scan_fn=hybrid.ascan` for native async hybrid/LLM scans (no thread wrapper).
- `ingest_document_async` and `trace_ingest_screen_async` mirror the sync ingest path for FastAPI/async services.

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

guard = ProtectRAGFilter()  # parallel screening across docs (tune with max_workers=1 for sequential)
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
from protectrag.integrations.fastapi import create_screening_middleware, screen_text_dependency_async

app = FastAPI()
app.add_middleware(create_screening_middleware(paths=["/api/ingest"]))
# Async routes: use screen_text_dependency_async(async_scan_fn=hybrid.ascan) with Depends(...)
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

`RunContext` also supports **`log_sample_rate_block`** and **`log_sample_rate_warn`** (0.0–1.0) to sample high-volume structured logs; metrics and callbacks are always recorded.

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
| `OPENAI_API_KEY` | — | Required for OpenAI-compatible Chat Completions |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible server |
| `PROTECTRAG_LLM_PROVIDER` | `auto` | `openai` / `openai_compatible` forces Chat Completions; `anthropic` / `claude` uses Anthropic’s **Messages** API (`/v1/messages`). With `auto`, Anthropic is selected if `OPENAI_BASE_URL` contains `anthropic.com`. |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic (unless you set `LLMScanConfig.api_key` in code) |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1` | Anthropic API base (used when `PROTECTRAG_LLM_PROVIDER=anthropic`) |
| `PROTECTRAG_LLM_MODEL` | `gpt-4o-mini` (OpenAI) or `claude-3-5-haiku-20241022` (Anthropic) | Classification model id for the chosen provider |
| `PROTECTRAG_HYBRID_MAX_RECALL` | *(unset)* | If `true` / `1` / `yes` / `on`, **LLM runs on every chunk** (including heuristic-clean and heuristic-HIGH) — best chance to catch subtle RAG poisoning; **highest** cost/latency. Overrides the two flags below when set. |
| `PROTECTRAG_HYBRID_LLM_ALWAYS` | *(unset)* | If `true` / `1` / `yes` / `on`, `HybridPolicy.from_env()` sets **run LLM on every chunk** (even when heuristics are clean) — higher cost/latency, fewer paraphrase misses |
| `PROTECTRAG_HYBRID_SKIP_LLM_IF_HIGH` | *(unset)* | If `false` / `0` / `no` / `off`, hybrid still calls the LLM when heuristics are already HIGH (unusual) |

`LLMScanConfig` also exposes **`llm_provider`**, **`anthropic_version`** (for the `anthropic-version` header), **`http_max_connections`** / **`http_max_keepalive_connections`** for httpx pool sizing under load.

**Claude (Anthropic) in code:** `LLMScanConfig(llm_provider="anthropic", api_key="...", model="claude-3-5-haiku-20241022")` or `LLMScanner.from_env()` with `PROTECTRAG_LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`. The model must return a JSON object with `severity` / `confidence` / `brief` (same schema as OpenAI); the system prompt already asks for that.

Use **`HybridScanner(llm, policy=HybridPolicy.from_env())`** to pick up the hybrid env vars.

**Hybrid behavior:** by default the LLM is **not** called when heuristics return a clean `NONE` result, so anything rules miss is never shown to the model. For **strongest** coverage against disguised or novel attacks, set `PROTECTRAG_HYBRID_MAX_RECALL=true` (or `HybridPolicy.max_recall()` in code) so **every** chunk is classified by the model. Lighter option: `PROTECTRAG_HYBRID_LLM_ALWAYS=true` (LLM on clean chunks; heuristic-HIGH still skips the LLM by default). When heuristics are already **HIGH**, the LLM is skipped unless you use max-recall or `PROTECTRAG_HYBRID_SKIP_LLM_IF_HIGH=false`.

Heuristic-only usage needs **none** of these.

---

## Concepts & full API

| Export | Role |
|--------|------|
| `scan_document_for_injection` | Heuristic scan → `DocumentScanResult` |
| `ingest_document` | Scan + policy + logs/metrics/callbacks |
| `ingest_document_async` | Same pipeline for async classifiers / FastAPI |
| `screen_retrieved_chunks` | Retrieval-time filtering (`max_workers` for threads) |
| `screen_retrieved_chunks_async` | Async concurrent retrieval screening |
| `async_scan`, `async_scan_batch`, `AsyncScanFn` | Async + batch (`async_scan_fn`, `batch_chunk_size`) |
| `LLMScanner`, `HybridScanner` | LLM / hybrid (`ascan`, `shared_cache`, httpx limits) |
| `RedisLLMClassificationCache`, `LLMClassificationCache` | Cross-replica LLM result cache |
| `with_retry`, `with_retry_async`, `RetryConfig` | Sync / async LLM retry + heuristic fallback |
| `CallbackRegistry` | `on_block` / `on_warn` / `on_allow` hooks |
| `load_golden_v1`, `run_eval_dataset` | Golden dataset + eval reports |
| `RunContext`, `InMemoryMetrics` | Run correlation + metrics |
| `configure_logging`, `emit_ingest_event` | Structured JSON logging |
| `trace_ingest_screen`, `trace_ingest_screen_async` | Latency + optional OTel spans |
| `span_attributes_for_ingest_scan` | OTel span attributes |
| `ProtectRAGFilter` | LangChain document transformer |
| `ProtectRAGPostprocessor` | LlamaIndex node postprocessor |
| `create_screening_middleware`, `screen_text_dependency_async` | FastAPI middleware + async Depends |

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
- **Default hybrid mode is not a full second line of defense:** if heuristics say `NONE`, the LLM is skipped, so sophisticated malicious text that evades regex never reaches the classifier. Use **`PROTECTRAG_HYBRID_MAX_RECALL=true`** (or `HybridPolicy.max_recall()`) when you need the model on every chunk; expect higher cost and some false positives.
- The LLM classifier can still miss adversarial or deeply hidden instructions; it is **not** a cryptographic guarantee—combine with trusted data sources, access control, and human review for high-risk corpora.
- Long texts are truncated (head + tail) before sending to the LLM.
- No built-in web dashboard — use notebooks, Grafana, or OTel backends.

---

## Community & support

- **Bug reports & feature requests:** [GitHub Issues](https://github.com/Ritvik777/ProtectRAG/issues)
- **Usage questions:** Open an issue with your Python version, `protectrag` version, and a minimal example if possible
- **Project home & source:** [github.com/Ritvik777/ProtectRAG](https://github.com/Ritvik777/ProtectRAG)

---

## Contributing

We welcome contributions: bug fixes, documentation improvements, tests, heuristic tuning (with eval impact), and careful API extensions.

Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** for local setup, how to run tests, expectations for pull requests, and where code lives in the tree.

---

## Security

To report a **security vulnerability** privately, follow **[SECURITY.md](SECURITY.md)** (use GitHub Security Advisories rather than a public issue when disclosure could harm users).

---

## Development

```bash
git clone https://github.com/Ritvik777/ProtectRAG.git
cd ProtectRAG
python -m venv .venv && source .venv/bin/activate  # optional
pip install -e ".[dev]"
pytest tests/ -v          # 60+ tests
python -m build           # build wheel + sdist
```

Full contributor workflow, PR checklist, and conventions: **[CONTRIBUTING.md](CONTRIBUTING.md)**.

---

## Repository layout

| Path | Purpose |
|------|---------|
| [`src/protectrag/`](src/protectrag/) | Library package (`scanner`, `ingest`, `llm`, `async_api`, `retrieval`, `integrations`, …) |
| [`tests/`](tests/) | Pytest suite |
| [`src/protectrag/data/`](src/protectrag/data/) | Bundled data (e.g. golden eval set) |
| [`notebooks/`](notebooks/) | Example Jupyter workflow |
| [`docs/`](docs/) | Extra documentation pointers |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute |
| [`SECURITY.md`](SECURITY.md) | Vulnerability reporting |

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
