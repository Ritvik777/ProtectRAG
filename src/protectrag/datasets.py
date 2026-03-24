"""Load the bundled golden dataset as EvalCase rows."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from protectrag.evals import EvalCase, GroundTruth

_LABEL_MAP = {"injection": GroundTruth.INJECTION, "clean": GroundTruth.CLEAN}
_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "datasets"


def load_golden_v1(path: str | Path | None = None) -> list[EvalCase]:
    """
    Load the bundled ``datasets/golden_v1.json`` as a list of :class:`EvalCase`.

    Pass ``path`` to override the file location.
    """
    p = Path(path) if path else _DATASET_DIR / "golden_v1.json"
    raw: list[dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for row in raw:
        gt = _LABEL_MAP.get(row.get("label", "unknown"), GroundTruth.UNKNOWN)
        tags = tuple(row.get("tags", ()))
        cases.append(EvalCase(id=row["id"], text=row["text"], ground_truth=gt, tags=tags))
    return cases
