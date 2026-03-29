"""Run / trace context for correlating logs, metrics, and experiments (Galileo-style runs)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunContext:
    """
    Correlate ingest events with a logical run (offline eval batch, CI job, or deploy).

    Similar in spirit to experiment/run IDs in Galileo Evaluate or Phoenix projects:
    same fields can be attached to OTel resource attributes or log pipelines.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project: str = "default"
    environment: str = "dev"  # dev | staging | prod
    dataset_name: str | None = None  # optional eval dataset name
    dataset_version: str | None = None
    started_at: str = field(default_factory=_utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)
    # Structured log sampling (0.0–1.0). Metrics and callbacks are unaffected.
    log_sample_rate_block: float = 1.0
    log_sample_rate_warn: float = 1.0

    def to_log_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "run_id": self.run_id,
            "project": self.project,
            "environment": self.environment,
            "started_at": self.started_at,
        }
        if self.dataset_name:
            out["dataset_name"] = self.dataset_name
        if self.dataset_version:
            out["dataset_version"] = self.dataset_version
        if self.extra:
            out.update(self.extra)
        return out
