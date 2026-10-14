"""Shared configuration, result type and comparison helpers for the model validation CLI."""

from __future__ import annotations

import math
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

API_BASE = 'http://localhost:4603'
EXPECTED_FILE = Path(__file__).resolve().parents[2] / 'tests' / 'expected_results.json'

# Tolerances — TensorRT rebuilds may shift values slightly
BOX_TOL = 0.03  # normalized coordinate tolerance
CONF_TOL = 0.08  # confidence score tolerance
NORM_TOL = 0.15  # embedding norm tolerance
COSINE_TOL = 0.98  # minimum cosine similarity for embeddings


# =============================================================================
# Utilities
# =============================================================================


class ValidationResult:
    def __init__(self, name: str):
        self.name = name
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, label: str, passed: bool, detail: str = '') -> bool:
        self.checks.append((label, passed, detail))
        return passed

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    @property
    def num_passed(self) -> int:
        return sum(1 for _, ok, _ in self.checks if ok)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def boxes_match(expected: list[float], actual: list[float], tol: float = BOX_TOL) -> bool:
    """Check if two bounding boxes are within tolerance."""
    if len(expected) != len(actual):
        return False
    return all(abs(e - a) <= tol for e, a in zip(expected, actual, strict=False))


def print_result(vr: ValidationResult, verbose: bool = False) -> None:
    """Print validation result for a modality."""
    icon = '\033[92m✓\033[0m' if vr.passed else '\033[91m✗\033[0m'
    status = 'PASS' if vr.passed else 'FAIL'
    color = '\033[92m' if vr.passed else '\033[91m'
    reset = '\033[0m'
    print(f'  {icon} {color}{status}{reset}  {vr.name} ({vr.num_passed}/{len(vr.checks)} checks)')

    if verbose or not vr.passed:
        for label, ok, detail in vr.checks:
            check_icon = '  ✓' if ok else '  ✗'
            check_color = '\033[92m' if ok else '\033[91m'
            msg = f'    {check_color}{check_icon}\033[0m {label}'
            if detail:
                msg += f' — {detail}'
