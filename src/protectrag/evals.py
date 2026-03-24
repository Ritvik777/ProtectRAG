"""
Offline evaluation datasets and run reports (Galileo-style experiments on golden sets).

Use labeled cases to track precision/recall of your screening policy; unlabeled cases
still produce pass-rate style stats for regression runs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from protectrag.scanner import DocumentScanResult, InjectionSeverity


class GroundTruth(str, Enum):
    """Whether the chunk truly contains prompt-injection intent (human or oracle label)."""

    CLEAN = "clean"
    INJECTION = "injection"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvalCase:
    """One row in a golden dataset."""

    id: str
    text: str
    ground_truth: GroundTruth = GroundTruth.UNKNOWN
    tags: tuple[str, ...] = ()


@dataclass
class EvalCaseResult:
    case_id: str
    ground_truth: GroundTruth
    predicted_alert: bool
    severity: InjectionSeverity
    score: float
    detector: str
    tags: tuple[str, ...]


@dataclass
class EvalReport:
    """Aggregate report for a single evaluation run."""

    run_id: str
    project: str
    n_cases: int
    n_labeled: int
    # Confusion matrix for alert (positive = treat as risky) vs injection present
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    # Unlabeled: how often scanner fired
    unlabeled_alerts: int = 0
    case_results: list[EvalCaseResult] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        if self.true_positives + self.false_positives == 0:
            return None
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float | None:
        if self.true_positives + self.false_negatives == 0:
            return None
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def accuracy(self) -> float | None:
        denom = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if denom == 0:
            return None
        return (self.true_positives + self.true_negatives) / denom

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project": self.project,
            "n_cases": self.n_cases,
            "n_labeled": self.n_labeled,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "unlabeled_alerts": self.unlabeled_alerts,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
        }


def _predict_alert(
    result: DocumentScanResult,
    *,
    alert_if_severity: InjectionSeverity = InjectionSeverity.MEDIUM,
) -> bool:
    return result.severity >= alert_if_severity


def run_eval_dataset(
    cases: list[EvalCase],
    *,
    classify: Callable[[str, str], DocumentScanResult],
    run_id: str = "eval",
    project: str = "protectrag",
    alert_if_severity: InjectionSeverity = InjectionSeverity.MEDIUM,
) -> EvalReport:
    """
    Run every case through ``classify(text, case_id)`` and aggregate metrics.

    ``classify`` should be your production path, e.g. hybrid or heuristic scan.
    """
    report = EvalReport(run_id=run_id, project=project, n_cases=len(cases), n_labeled=0)
    labeled = 0
    for c in cases:
        r = classify(c.text, c.id)
        pred = _predict_alert(r, alert_if_severity=alert_if_severity)
        if c.ground_truth != GroundTruth.UNKNOWN:
            labeled += 1
            is_inj = c.ground_truth == GroundTruth.INJECTION
            if is_inj and pred:
                report.true_positives += 1
            elif is_inj and not pred:
                report.false_negatives += 1
            elif not is_inj and pred:
                report.false_positives += 1
            else:
                report.true_negatives += 1
        else:
            if pred:
                report.unlabeled_alerts += 1
        report.case_results.append(
            EvalCaseResult(
                case_id=c.id,
                ground_truth=c.ground_truth,
                predicted_alert=pred,
                severity=r.severity,
                score=r.score,
                detector=r.detector,
                tags=c.tags,
            )
        )
    report.n_labeled = labeled
    return report
