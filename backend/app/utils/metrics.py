from __future__ import annotations

from prometheus_client import Counter


rag_parse_counter = Counter(
    "rag_parse_attempts_total",
    "Total RAG parsing attempts",
    ["success", "fallback", "examples_count"],
)
