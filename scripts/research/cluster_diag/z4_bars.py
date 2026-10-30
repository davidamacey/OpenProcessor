"""The Z4 bars and the adoption rule (decision note section 0A.2), fixed before the set was scored.

Inputs are ``loop_sim.seeded_loop`` summaries (and ``gate_sim.seeded_runs`` for IVF without
recursion) read on the held-out half, fold 1, with the clean oracle labeler. Nothing here is tuned
after a result.
"""

from __future__ import annotations

from typing import Any


U2_BAR = 0.95  # B-a: propagated-label precision
IVF_FACTOR = 0.5  # B-b: decisions at most this share of IVF's
OPEN_BAR = 0.90  # B-c: precision on open-set items
MIN_OPEN_PROPAGATED = 30  # B-c is not evaluable below this many propagated open-set items (mean)

PASS, FAIL, INCONCLUSIVE = 'PASS', 'FAIL', 'INCONCLUSIVE'
KEY_ARMS = ('key', 'key_guard')  # the order breaks a tie toward the simpler arm
ARMS = (*KEY_ARMS, 'within_key')


def _mean(block: dict[str, Any] | None, key: str) -> float | None:
    cell = None if block is None else block.get(key)
    return None if cell is None else float(cell['mean'])


def ivf_reference(ivf_loop: dict[str, Any], ivf_gate: dict[str, Any]) -> float | None:
    """The incumbent's decisions per item: the cheaper of IVF with the loop's recursion and IVF
    asked whole when it fails, so the bar is not flattered by the weaker comparator."""
    values = [v for v in (_mean(ivf_loop, 'u1'), _mean(ivf_gate, 'u1')) if v is not None]
    return min(values) if values else None


def arm_rows(arm: dict[str, Any], ivf_u1: float | None) -> list[dict[str, Any]]:
    """B-a to B-c for one arm as rows with PASS, FAIL or INCONCLUSIVE and the limit that applies."""
    u2, u1 = _mean(arm, 'u2'), _mean(arm, 'u1')
    rows: list[dict[str, Any]] = []

    def row(bar_id: str, bar: float | None, value: float | None, result: str, note: str) -> None:
        rows.append({'id': bar_id, 'bar': bar, 'value': value, 'result': result, 'note': note})

    if u2 is None:
        row('B-a', U2_BAR, None, INCONCLUSIVE, 'nothing was propagated')
    else:
        row('B-a', U2_BAR, u2, PASS if u2 >= U2_BAR else FAIL, '')
    if u1 is None or ivf_u1 is None:
        row('B-b', None, u1, INCONCLUSIVE, 'u1 missing')
    else:
        bar = IVF_FACTOR * ivf_u1
        row('B-b', bar, u1, PASS if u1 <= bar else FAIL, f'ivf u1 {ivf_u1:.4f}')
    open_u2 = _mean(arm, 'u2_open')
    open_n = _mean(arm, 'open_propagated') or 0.0
    if open_u2 is None or open_n < MIN_OPEN_PROPAGATED:
        row(
            'B-c',
            OPEN_BAR,
            open_u2,
            INCONCLUSIVE,
            f'{open_n:.1f} open-set items propagated per run, fewer than {MIN_OPEN_PROPAGATED}',
        )
    else:
        row(
            'B-c',
            OPEN_BAR,
            open_u2,
            PASS if open_u2 >= OPEN_BAR else FAIL,
            f'{open_n:.1f} open-set items propagated per run',
        )
    return rows


def verdict(rows: list[dict[str, Any]]) -> str:
    """FAIL if any bar fails, else INCONCLUSIVE if any is, else PASS."""
    results = {r['result'] for r in rows}
    if FAIL in results:
        return FAIL
    return INCONCLUSIVE if INCONCLUSIVE in results else PASS


def adoption(verdicts: dict[str, str], u1: dict[str, float | None]) -> dict[str, Any]:
    """The registered rule: key roots (with or without the guard, the cheaper one that passes);
    else within-key roots if they pass; else IVF stays. An arm that did not PASS is never
    adopted, whether it failed or could not be read."""
    passing = [a for a in KEY_ARMS if verdicts.get(a) == PASS and u1.get(a) is not None]
    if passing:
        pick = min(passing, key=lambda a: (u1[a], KEY_ARMS.index(a)))
        return {'adopt': pick, 'reason': 'key roots pass'}
    if verdicts.get('within_key') == PASS:
        return {'adopt': 'within_key', 'reason': 'no key arm passes, within-key roots pass'}
    unreadable = [a for a in ARMS if verdicts.get(a) == INCONCLUSIVE]
    return {
        'adopt': 'ivf',
        'reason': 'no arm passes',
        'inconclusive_arms': unreadable,
    }


def overall(verdicts: dict[str, str]) -> str:
    """PASS if some arm passes; FAIL if every arm fails; else INCONCLUSIVE."""
    values = [verdicts[a] for a in ARMS]
    if PASS in values:
        return PASS
    return FAIL if all(v == FAIL for v in values) else INCONCLUSIVE
