/**
 * Static Q&A "chat" for GitHub Pages — no server; pattern + keyword matching.
 */
(function () {
  const KNOWLEDGE = [
    {
      q: ["what is protectrag", "what is this", "introduce", "overview"],
      a: `ProtectRAG is a <strong>Python library</strong> that screens text before it goes into a RAG vector store (and optionally <strong>after retrieval</strong>) for <strong>prompt injection</strong> patterns. It applies <code>ALLOW</code> / <code>WARN</code> / <code>BLOCK</code> policies and plugs into logs, metrics, callbacks, and OpenTelemetry. It does <strong>not</strong> run a hosted service or talk to your vector DB — you call it from your pipeline.`,
    },
    {
      q: ["prompt injection", "why", "attack"],
      a: `Documents indexed for RAG can hide instructions like “ignore previous instructions” or fake system markers. When that chunk is retrieved, the model may follow those instructions. ProtectRAG scores risk with <strong>13+ heuristic rule families</strong> and an optional <strong>LLM classifier</strong> so you can block or warn before indexing or before the LLM sees the context.`,
    },
    {
      q: ["install", "pip", "pypi"],
      a: `Core (no required deps): <code>pip install protectrag</code><br><br>Extras: <code>pip install "protectrag[llm]"</code> (httpx), <code>[langchain]</code>, <code>[llamaindex]</code>, <code>[fastapi]</code>, <code>[otel]</code>, <code>[redis]</code> for shared LLM cache. Requires <strong>Python 3.10+</strong>.`,
    },
    {
      q: ["heuristic", "rules", "regex", "without llm", "free"],
      a: `The default path is <strong>heuristic-only</strong>: compiled regexes for instruction overrides, multilingual patterns, role delimiters, exfiltration, encoding tricks, Unicode tricks, HTML/markdown, indirect injection, payload splitting, and more. No API key needed. Use <code>scan_document_for_injection</code> or <code>ingest_document</code> with the default scanner.`,
    },
    {
      q: ["hybrid", "llm", "openai", "gpt"],
      a: `Use <code>LLMScanner</code> with an OpenAI-compatible API (<code>OPENAI_API_KEY</code>, optional <code>OPENAI_BASE_URL</code>). <code>HybridScanner</code> runs heuristics first and can <strong>skip the LLM</strong> when text is clean or already obviously bad — saving latency and cost. Async: <code>ascan</code> / <code>ingest_document_async</code>.`,
    },
    {
      q: ["retrieval", "after retrieve", "chunks"],
      a: `Use <code>screen_retrieved_chunks</code> (sync, optional thread pool) or <code>screen_retrieved_chunks_async</code> (bounded concurrency) on <code>RetrievedChunk</code> lists. Default <code>block_on</code> is often stricter at retrieval time because text is about to reach the model.`,
    },
    {
      q: ["fastapi", "middleware", "api"],
      a: `Optional extra: <code>protectrag.integrations.fastapi</code> — middleware for JSON bodies and dependencies including <code>screen_text_dependency_async</code> for async hybrid/LLM scans without blocking the event loop.`,
    },
    {
      q: ["vector", "pinecone", "database", "store"],
      a: `ProtectRAG only sees <strong>strings</strong>. It works with any vector store (Pinecone, Qdrant, pgvector, etc.): call it in your ingest worker before embed/upsert and/or on retrieved chunks before building the LLM context.`,
    },
    {
      q: ["eval", "test", "dataset", "golden"],
      a: `Ships with a <strong>golden dataset</strong> (<code>load_golden_v1</code>) and <code>run_eval_dataset</code> for precision/recall-style reports in CI when you change rules or models.`,
    },
    {
      q: ["contribute", "contributing", "pr"],
      a: `See <a href="https://github.com/Ritvik777/ProtectRAG/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener">CONTRIBUTING.md</a> on GitHub: dev setup, pytest, and PR expectations.`,
    },
    {
      q: ["security", "vulnerability", "report"],
      a: `For sensitive issues, follow <a href="https://github.com/Ritvik777/ProtectRAG/blob/main/SECURITY.md" target="_blank" rel="noopener">SECURITY.md</a> and use GitHub Security Advisories instead of public issues.`,
    },
    {
      q: ["redis", "cache", "scale"],
      a: `Optional <code>RedisLLMClassificationCache</code> lets multiple workers share LLM classification results (same body → fewer API calls). Install <code>protectrag[redis]</code> and pass <code>shared_cache=...</code> to <code>LLMScanner</code>.`,
    },
  ];

  function normalize(s) {
    return s.toLowerCase().trim().replace(/\s+/g, " ");
  }

  function matchAnswer(text) {
    const n = normalize(text);
    if (!n) return null;

    for (const item of KNOWLEDGE) {
      for (const phrase of item.q) {
        if (n.includes(phrase) || phrase.split(" ").every((w) => w.length > 2 && n.includes(w))) {
          return item.a;
        }
      }
    }

    return (
      `I don’t have a scripted answer for that on this static page. Try the <strong>suggested questions</strong> below, or ask on ` +
      `<a href="https://github.com/Ritvik777/ProtectRAG/issues" target="_blank" rel="noopener">GitHub Issues</a> / read the ` +
      `<a href="https://github.com/Ritvik777/ProtectRAG/blob/main/README.md" target="_blank" rel="noopener">README</a>.`
    );
  }

  function appendMessage(role, html) {
    const log = document.getElementById("chat-log");
    if (!log) return;
    const div = document.createElement("div");
    div.className = "msg " + (role === "user" ? "msg-user" : "msg-bot");
    div.innerHTML = html;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function sendFromInput() {
    const input = document.getElementById("chat-input");
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    appendMessage("user", escapeHtml(text));
    const answer = matchAnswer(text);
    setTimeout(() => appendMessage("bot", answer), 280);
    input.value = "";
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("chat-input");
    const send = document.getElementById("chat-send");

    if (send) send.addEventListener("click", sendFromInput);
    if (input) {
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendFromInput();
        }
      });
    }

    document.querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const inputEl = document.getElementById("chat-input");
        if (inputEl) inputEl.value = chip.dataset.q || chip.textContent;
        sendFromInput();
      });
    });
  });
})();
