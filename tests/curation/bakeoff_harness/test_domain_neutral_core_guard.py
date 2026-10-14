"""Guards the harness core against domain vocabulary creeping back in."""

from __future__ import annotations

import re

import pytest

from curation.bakeoff_harness._helpers import HARNESS, REPO_ROOT, _FakeDetector
from scripts.curation.bakeoff.backends import registry


# --- domain-neutral core guard ---------------------------------------------------

# Domain content (profiles, baselines, converters, backends for one domain's
# public models) lives only under the opt-in examples/ tree. The router is
# still being rewritten (its v1 request model still names legacy
# backends) and joins this scan once that lands; the two related service
# modules are scanned as soon as they exist.
_DOMAIN_WORDS = re.compile(r'(?i)plate|lpr|lpdnet|vehicle|open[_-]?image')
_BAKEOFF_API_FILES = (
    'src/routers/curation/bakeoff.py',
    *(f'src/schemas/bakeoff/{m}.py' for m in ['datasets', 'job_spec', 'models', 'results', 'run']),
    'src/services/curation/eval_datasets.py',
    'src/services/curation/bakeoff_jobs.py',
)


def test_harness_core_has_no_domain_vocabulary() -> None:
    files = [p for p in sorted(HARNESS.rglob('*')) if p.suffix in {'.py', '.json', '.txt', '.md'}]
    files += [REPO_ROOT / f for f in _BAKEOFF_API_FILES]
    offenders: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if _DOMAIN_WORDS.search(line):
                offenders.append(f'{rel}:{n}: {line.strip()}')
        if _DOMAIN_WORDS.search(rel):
            offenders.append(rel)
    assert not offenders, 'domain vocabulary in the generic harness core:\n' + '\n'.join(offenders)


def test_builtin_backends_are_generic() -> None:
    """Before any plugin module is imported, only the five generic backends exist."""
    import subprocess
    import sys

    code = (
        'from scripts.curation.bakeoff.backends import registry; '
        'print(",".join(sorted(registry.registered_backends())))'
    )
    proc = subprocess.run(
        [sys.executable, '-c', code], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    builtins = {'ultralytics', 'triton', 'two-stage', 'onnxruntime', 'coreml'}
    assert set(proc.stdout.strip().split(',')) == builtins
    from scripts.curation.bakeoff.profile import BACKENDS

    assert set(BACKENDS) == builtins
    with pytest.raises(SystemExit, match='unknown bake-off backend'):
        registry.get_backend('no-such-backend')
    with pytest.raises(ValueError, match='already registered'):
        registry.register_backend('ultralytics', lambda _args: _FakeDetector('x'))


def test_paper_modules_are_not_in_the_harness() -> None:
    for name in ('lean_candidates', 'deskew_prototype', 'dedup_sweep', 'paper_numbers'):
        assert not (HARNESS / f'{name}.py').exists(), name
