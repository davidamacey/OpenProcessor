#!/usr/bin/env python3
"""What the native frame engine must equal in settings, metrics and lane arithmetic (plan 17 N2).

Writes under ``native/frame_engine/tests/golden/``:

* ``settings_cases.json``: environments run through the Python engine's real
  ``EngineSettings.from_env`` (every ``OP_FRAME_ENGINE_*`` it reads at its default, at both ends
  of its range, one step outside, malformed, empty) with what it answered, the settings or the
  refusal text. The Rust test replays them. The variables and ranges are read from the source of
  ``from_env`` so a setting added there is replayed without editing this script.
* ``metrics_families.json``: every metric family the Python engine registers (name, type, label
  names, bucket bounds for a histogram) taken from its registry, and the reviewed differences.
* ``lane_budget.json``: cases of ``frame_engine_lane_budget.lanes_that_fit``.

``--check`` regenerates into memory and compares with the committed files.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / 'native' / 'frame_engine' / 'tests' / 'golden'
SCRIPT = 'scripts/codegen/frame_engine_parity_goldens.py'
PREFIX = 'OP_FRAME_ENGINE_'

#: Settings the Python engine reads that the native engine answers differently, each with why.
SETTINGS_DIFFERENCES = {
    'OP_FRAME_ENGINE_SWITCH_INTERVAL_MS': (
        "tunes Python's interpreter lock; the native engine has none, so a start with it set is "
        'refused (plan 17 section 1.1)'
    ),
    'OP_FRAME_ENGINE_LANES': (
        'the default is the physical cores, lowered to what fits in the card (plan 17 section '
        '2.1 and N2); a value that is set means the same and has the same range'
    ),
}
#: Settings only the native engine reads (documented in env.template).
NATIVE_ONLY_SETTINGS = {
    'OP_FRAME_ENGINE_EXPECT_BUS_ID': (
        'refuse the start on any card but this PCI bus id (a host run names its card by it)'
    ),
    'OP_FRAME_ENGINE_LANDING_ROOT': (
        'the directory a landing/... key is read under; the Python engine has no read by key'
    ),
    'OP_FRAME_ENGINE_SOURCE_ROOT': (
        'the directory a source/... key is read under; the Python engine has no read by key'
    ),
}
#: A value the native engine must refuse, for the native-only settings whose value has a form.
#: (A root is any path; the start checks it is a directory, the settings read does not.)
NATIVE_ONLY_REFUSED_VALUES = {'OP_FRAME_ENGINE_EXPECT_BUS_ID': 'card two'}
#: Fields of ``EngineSettings`` that are not settings of a deployment.
NOT_REPLAYED_FIELDS = ('ocr_dictionary', 'switch_interval_ms')

#: Metric families one engine has and the other must not, each with why (reviewed).
METRICS_NATIVE_ONLY = {
    'op_frame_engine_nvjpeg_in_use_bytes': (
        "gauge, device bytes of the decode lanes' nvJPEG buffers; the native engine counts them "
        'because they sit outside the pool cap (plan 17 section 2.3)'
    ),
    'op_frame_engine_nvjpeg_peak_bytes': (
        'gauge, the most of the above since start; the figure the lane count is sized from'
    ),
}
METRICS_PYTHON_ONLY: dict[str, str] = {}


def _from_env_calls() -> list[dict[str, Any]]:
    """The ``env_int`` and ``env_float`` calls of ``EngineSettings.from_env``: name, field,
    kind, low and high, read from the source."""
    tree = ast.parse((ROOT / 'frame_engine' / 'settings.py').read_text(encoding='utf-8'))
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in ('env_int', 'env_float') or len(node.args) != 4:
            continue
        name, default, lo, hi = node.args
        if not isinstance(name, ast.Constant) or not isinstance(default, ast.Attribute):
            continue
        found.append(
            {
                'name': name.value,
                'field': default.attr,
                'kind': 'int' if node.func.id == 'env_int' else 'float',
                'lo': ast.literal_eval(lo),
                'hi': ast.literal_eval(hi),
            }
        )
    return found


def _names_python_reads() -> list[str]:
    """Every ``OP_FRAME_ENGINE_*`` name ``frame_engine/settings.py`` hands to one of its
    ``env_*`` readers, read from the source (the ranged ones and the text and boolean ones)."""
    tree = ast.parse((ROOT / 'frame_engine' / 'settings.py').read_text(encoding='utf-8'))
    names = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if not node.func.id.startswith('env_') or not node.args:
            continue
        first = node.args[0]
        if (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value.startswith(PREFIX)
        ):
            names.add(first.value)
    return sorted(names)


#: Settings that must be set beside a boundary value for it to be accepted (the relations).
COMPANIONS = {
    'OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS': {'OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS': '600'},
    'OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS': {'OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS': '1'},
    'OP_FRAME_ENGINE_BUDGET_MIB': {'OP_FRAME_ENGINE_POOL_MIB': '262144'},
    'OP_FRAME_ENGINE_POOL_MIB': {'OP_FRAME_ENGINE_BUDGET_MIB': '64'},
}


def _number(value: float, kind: str) -> str:
    return str(int(value)) if kind == 'int' else repr(float(value))


def _env_cases() -> list[tuple[str, dict[str, str]]]:
    cases: list[tuple[str, dict[str, str]]] = [('defaults', {})]
    for call in _from_env_calls():
        name, kind, lo, hi = call['name'], call['kind'], call['lo'], call['hi']
        extra = COMPANIONS.get(name, {})
        for label, value in (
            ('low', _number(lo, kind)),
            ('high', _number(hi, kind)),
            ('below', _number(lo - 1, kind)),
            ('above', _number(hi + 1, kind)),
            ('word', 'many'),
            ('empty', ''),
            ('padded', f' {_number(lo, kind)} '),
            ('separator', '1_0' if kind == 'int' else '1_0.5'),
            ('double_separator', '1__0'),
        ):
            cases.append((f'{name} {label}', {**extra, name: value}))
        if kind == 'float':
            cases += [(f'{name} {w}', {name: w}) for w in ('nan', 'inf', '1e1')]
        else:
            cases.append((f'{name} fraction', {name: '1.5'}))
    cases += [
        ('transport body', {PREFIX + 'TRANSPORT': 'body'}),
        ('transport word', {PREFIX + 'TRANSPORT': 'shm'}),
        ('instance dashes', {PREFIX + 'INSTANCE': 'fe-1'}),
        ('instance space', {PREFIX + 'INSTANCE': 'fe 0'}),
        ('instance only dashes', {PREFIX + 'INSTANCE': '-'}),
        ('triton url', {PREFIX + 'TRITON_URL': 'triton-gpu2:8001'}),
        ('switch interval', {PREFIX + 'SWITCH_INTERVAL_MS': '0.1'}),
        ('budget above pool', {PREFIX + 'BUDGET_MIB': '8192'}),
        ('timeout above deadline', {PREFIX + 'TRITON_TIMEOUT_SECONDS': '90'}),
    ]
    cases += [
        (f'stage_sync {v}', {PREFIX + 'STAGE_SYNC': v})
        for v in ('1', 'TRUE', 'yes', 'On', '0', 'false', 'no', 'OFF', 'maybe', '2')
    ]
    return cases


def _run(env: dict[str, str]) -> dict[str, Any]:
    from frame_engine.settings import EngineSettings, SettingsError

    saved = {k: v for k, v in os.environ.items() if k.startswith(PREFIX)}
    for key in saved:
        del os.environ[key]
    os.environ.update(env)
    try:
        fields = asdict(EngineSettings.from_env())
        return {'ok': {k: v for k, v in fields.items() if k not in NOT_REPLAYED_FIELDS}}
    except SettingsError as e:
        return {'error': str(e)}
    finally:
        for key in env:
            os.environ.pop(key, None)
        os.environ.update(saved)


def render_settings() -> str:
    cases = [{'id': cid, 'env': env, **_run(env)} for cid, env in _env_cases()]
    body = {
        'generated_by': SCRIPT,
        'source': 'frame_engine/settings.py EngineSettings.from_env',
        'ranges': _from_env_calls(),
        'differences': SETTINGS_DIFFERENCES,
        'native_only': NATIVE_ONLY_SETTINGS,
        'native_only_refused_values': NATIVE_ONLY_REFUSED_VALUES,
        'python_names': _names_python_reads(),
        'cases': cases,
    }
    return json.dumps(body, sort_keys=True, separators=(',', ':')) + '\n'


def _family_rows(metrics: Any) -> list[dict[str, Any]]:
    """Name, type, labels and buckets of every family of the engine's registry. A family's
    labels are those its metric object was made with; the collector's families (the process
    series) have none."""
    from prometheus_client import Counter, Gauge, Histogram, Summary

    rows: dict[str, dict[str, Any]] = {}
    for obj in vars(metrics).values():
        if isinstance(obj, Counter | Gauge | Histogram | Summary):
            row: dict[str, Any] = {
                'name': obj._name,
                'type': obj._type,
                'labels': list(obj._labelnames),
            }
            if isinstance(obj, Histogram):
                row['buckets'] = list(obj._upper_bounds)
            rows[row['name']] = row
    for family in metrics.registry.collect():
        rows.setdefault(family.name, {'name': family.name, 'type': family.type, 'labels': []})
    return sorted(rows.values(), key=lambda r: r['name'])


def render_metrics() -> str:
    from frame_engine.stage_metrics import EngineMetrics

    body = {
        'generated_by': SCRIPT,
        'source': 'frame_engine/stage_metrics.py EngineMetrics',
        'families': _family_rows(EngineMetrics()),
        'native_only': METRICS_NATIVE_ONLY,
        'python_only': METRICS_PYTHON_ONLY,
    }
    return json.dumps(body, sort_keys=True, separators=(',', ':')) + '\n'


def render_lanes() -> str:
    import frame_engine_lane_budget as lb

    mib = lb.MIB
    cases = []
    for free_mib in (0, 100, 1000, 5600, 11764, 23552, 48000, 10**7):
        for pool_mib in (0, 4096):
            for mp in (1.0, 12.0, 20.0, 26.0, 40.0, 100.0):
                fit = lb.lanes_that_fit(free_mib * mib, pool_mib * mib, mp)
                cases.append(
                    {
                        'free_bytes': free_mib * mib,
                        'pool_bytes': pool_mib * mib,
                        'megapixels': mp,
                        'lanes': fit.lanes,
                        'fits': fit.fits,
                    }
                )
    sized = [
        {'max_megapixels': mp, 'sized': lb.sized_megapixels(mp)} for mp in (1.0, 12.0, 26.0, 100.0)
    ]
    # the `full` installer profile: the smallest card it is chosen for, its models, region and pool
    full = {'card_free_mib': 40960, 'models_mib': 17408, 'region_mib': 1024, 'pool_mib': 4096}
    body = {
        'generated_by': SCRIPT,
        'source': 'scripts/codegen/frame_engine_lane_budget.py',
        'sized': sized,
        'full_profile': {**full, 'lanes': lb.profile_lanes(True, *full.values())},
        'bytes_per_megapixel': lb.NVJPEG_BYTES_PER_MEGAPIXEL_PER_LANE,
        'headroom_percent': lb.HEADROOM_PERCENT,
        'sized_for_megapixels': lb.SIZED_FOR_MEGAPIXELS,
        'cases': cases,
    }
    return json.dumps(body, sort_keys=True, separators=(',', ':')) + '\n'


def render() -> dict[str, str]:
    return {
        'settings_cases.json': render_settings(),
        'metrics_families.json': render_metrics(),
        'lane_budget.json': render_lanes(),
    }


def main(argv: list[str] | None = None) -> int:
    import frame_engine_goldens

    return int(frame_engine_goldens.main(argv, render, SCRIPT, out=OUT))


if __name__ == '__main__':
    raise SystemExit(main())
