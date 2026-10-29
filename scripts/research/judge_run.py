"""Run loop for the visual judge: write-ahead budget, verdict cache and raw JSONL records."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from scripts.research.judge_protocol import (
    PROMPT_VERSION,
    JudgeRequest,
    RawReply,
    VerdictError,
    assert_may_send,
    estimate_tokens,
    parse_verdict,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class BudgetExceededError(RuntimeError):
    """A call would pass a hard cap; nothing was sent."""


class Backend(Protocol):
    hosted: bool
    model: str
    name: str

    def judge(self, request: JudgeRequest) -> RawReply | None: ...


class Budget:
    """Hard caps on calls and input tokens, reserved on disk before each call."""

    def __init__(self, path: Path, *, max_calls: int, max_input_tokens: int) -> None:
        self.path = path
        self.max_calls = max_calls
        self.max_input_tokens = max_input_tokens
        self.reserved: dict[str, int] = {}
        if path.is_file():
            self.reserved = dict(json.loads(path.read_text()).get('reserved', {}))

    @property
    def calls(self) -> int:
        return len(self.reserved)

    @property
    def input_tokens(self) -> int:
        return sum(self.reserved.values())

    def reserve(self, key: str, est_tokens: int) -> None:
        """Idempotent per key; raises BudgetExceededError before the caller may make the call."""
        if key in self.reserved:
            return
        if self.calls + 1 > self.max_calls:
            raise BudgetExceededError(f'max_calls {self.max_calls} reached')
        if self.input_tokens + est_tokens > self.max_input_tokens:
            raise BudgetExceededError(f'max_input_tokens {self.max_input_tokens} would be passed')
        self.reserved[key] = est_tokens
        self._flush()

    def settle(self, key: str, actual_tokens: int | None) -> None:
        if actual_tokens is not None and key in self.reserved:
            self.reserved[key] = actual_tokens
            self._flush()

    def summary(self) -> dict[str, int]:
        return {
            'max_calls': self.max_calls,
            'max_input_tokens': self.max_input_tokens,
            'calls_reserved': self.calls,
            'input_tokens_reserved': self.input_tokens,
        }

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        with tmp.open('w') as fh:
            json.dump({'reserved': self.reserved}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(self.path)


class VerdictCache:
    """Verdicts keyed by (sheet PNG hash, model, prompt version)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(png_sha256: str, model: str, prompt_version: str) -> str:
        return hashlib.sha256(f'{png_sha256}|{model}|{prompt_version}'.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.directory / f'{key}.json'
        return json.loads(path.read_text()) if path.is_file() else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f'{key}.json').write_text(json.dumps(record, sort_keys=True))


@dataclass
class RunOutcome:
    judged: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)


def _record(
    run_id: str,
    backend: Backend,
    req: JudgeRequest,
    raw: RawReply,
    verdict: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    m = req.manifest
    return {
        'run_id': run_id,
        'model': backend.model,
        'backend': backend.name,
        'prompt_version': PROMPT_VERSION,
        'unit': m['layout'],
        'kind': m['kind'],
        'sheet_id': m['sheet_id'],
        'png_sha256': m['png_sha256'],
        'verdict': verdict,
        'odd_cells': (verdict or {}).get('odd_cells'),
        'name': (verdict or {}).get('name'),
        'unsure': _unsure(verdict),
        'input_tokens': raw.input_tokens,
        'output_tokens': raw.output_tokens,
        'latency_ms': raw.latency_ms,
        'request_id': raw.request_id,
        'cached': False,
        'error': error,
        'raw_text': raw.text if error else None,
    }


def _unsure(verdict: dict[str, Any] | None) -> bool | int | None:
    if verdict is None:
        return None
    if 'rows' in verdict:
        return int(sum(r['answer'] == 'unsure' for r in verdict['rows']))
    if 'confidence' in verdict:
        return bool(verdict['confidence'] == 'low')
    return None


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as fh:
        fh.write(json.dumps(row, sort_keys=True) + '\n')


def run_requests(
    requests: Sequence[JudgeRequest],
    backend: Backend,
    budget: Budget,
    cache: VerdictCache | None,
    out_path: Path,
    *,
    run_id: str,
    header: dict[str, Any] | None = None,
    est_tokens: int | None = None,
) -> RunOutcome:
    """Judge each sheet once; raises PrivacyRefusalError or BudgetExceededError before the offending call."""
    _append(
        out_path,
        {
            'run_id': run_id,
            'model': backend.model,
            'backend': backend.name,
            'prompt_version': PROMPT_VERSION,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'budget': budget.summary(),
            **(header or {}),
        },
    )
    out = RunOutcome()
    for req in requests:
        _one(req, backend, budget, cache, out_path, run_id, est_tokens, out)
    return out


def _one(
    req: JudgeRequest,
    backend: Backend,
    budget: Budget,
    cache: VerdictCache | None,
    out_path: Path,
    run_id: str,
    est_tokens: int | None,
    out: RunOutcome,
) -> None:
    m = req.manifest
    sheet_id = m['sheet_id']
    ckey = VerdictCache.key(m['png_sha256'], backend.model, PROMPT_VERSION)
    hit = cache.get(ckey) if cache else None
    if hit is not None:
        _append(out_path, {**hit, 'run_id': run_id, 'cached': True})
        out.cached.append(sheet_id)
        return
    assert_may_send(m['kind'], hosted=backend.hosted)
    budget.reserve(ckey, est_tokens if est_tokens is not None else estimate_tokens(req.png))
    raw = backend.judge(req)
    if raw is None:
        out.pending.append(sheet_id)
        return
    budget.settle(ckey, raw.input_tokens)
    try:
        verdict, error = parse_verdict(m, raw.text), None
    except VerdictError as exc:
        verdict, error = None, str(exc)
        out.invalid.append(sheet_id)
    record = _record(run_id, backend, req, raw, verdict, error)
    _append(out_path, record)
    if cache and verdict is not None:
        cache.put(ckey, record)
    out.judged.append(sheet_id)
