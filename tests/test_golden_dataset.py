"""Tests that load and run the golden dataset."""

from __future__ import annotations

from protectrag.datasets import load_golden_v1
from protectrag.evals import GroundTruth, run_eval_dataset
from protectrag.scanner import scan_document_for_injection


def test_golden_dataset_loads() -> None:
    cases = load_golden_v1()
    assert len(cases) == 100
    injections = [c for c in cases if c.ground_truth == GroundTruth.INJECTION]
    cleans = [c for c in cases if c.ground_truth == GroundTruth.CLEAN]
    assert len(injections) == 50
    assert len(cleans) == 50


def test_golden_eval_reasonable_recall() -> None:
    cases = load_golden_v1()
    report = run_eval_dataset(
        cases,
        classify=lambda t, d: scan_document_for_injection(t, document_id=d),
        run_id="golden-test",
    )
    assert report.recall is not None
    assert report.recall >= 0.7, f"Recall too low: {report.recall}"
    assert report.precision is not None
    assert report.precision >= 0.7, f"Precision too low: {report.precision}"
