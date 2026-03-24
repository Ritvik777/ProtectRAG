"""Load the bundled golden dataset as EvalCase rows."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from protectrag.evals import EvalCase, GroundTruth

_LABEL_MAP = {"injection": GroundTruth.INJECTION, "clean": GroundTruth.CLEAN}


def _bundled_path() -> Path:
    """Resolve path to bundled data, works both editable and wheel installs."""
    return Path(__file__).resolve().parent / "data" / "golden_v1.json"


def load_golden_v1(path: str | Path | None = None) -> list[EvalCase]:
    """
    Load the bundled ``golden_v1.json`` as a list of :class:`EvalCase`.

    Pass ``path`` to override the file location.
    """
    p = Path(path) if path else _bundled_path()
    raw: list[dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for row in raw:
        gt = _LABEL_MAP.get(row.get("label", "unknown"), GroundTruth.UNKNOWN)
        tags = tuple(row.get("tags", ()))
        cases.append(EvalCase(id=row["id"], text=row["text"], ground_truth=gt, tags=tags))
    return cases
