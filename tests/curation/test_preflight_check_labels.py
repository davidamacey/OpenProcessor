"""Every training preflight row is served with a label (``PreflightCheck.label``): the name the
form shows instead of an id such as ``trainer_gpus``."""

from __future__ import annotations

import ast
from pathlib import Path

from src.services.training.preflight_checks import CHECK_LABELS, PreflightCheck, PreflightReport


TRAINING = Path(__file__).resolve().parents[2] / 'src' / 'services' / 'training'
SOURCES = (TRAINING / 'preflight_checks.py', TRAINING / 'preflight_run.py')


def _names_in_source() -> set[str]:
    """Every check name the two modules can emit: ``PreflightCheck(name='...')`` and the
    ``export_*`` names of the export readiness rows, which are passed as tuple keys."""
    names: set[str] = set()
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'PreflightCheck':
                names.update(
                    str(kw.value.value)
                    for kw in node.keywords
                    if kw.arg == 'name' and isinstance(kw.value, ast.Constant)
                )
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith('export_')
                and node.value in CHECK_LABELS
            ):
                names.add(node.value)
    return names


def test_every_check_the_code_emits_has_a_served_label() -> None:
    names = _names_in_source()
    assert {'class_balance', 'test_holdout', 'trainer_gpus', 'export_generation'} <= names
    assert names <= set(CHECK_LABELS), sorted(names - set(CHECK_LABELS))


def test_every_label_is_a_name_not_an_id() -> None:
    for name, label in CHECK_LABELS.items():
        assert label, name
        assert label[0].isupper(), name
        assert '_' not in label, name


def test_a_row_serves_its_label_beside_the_name() -> None:
    row = PreflightCheck(name='class_balance', severity='warn', message='4 classes under 500')
    served = row.model_dump(mode='json')
    assert served['label'] == 'Class balance'
    assert served['name'] == 'class_balance'
    report = PreflightReport(blocked=False, checks=[row])
    assert report.model_dump(mode='json')['checks'][0]['label'] == 'Class balance'
