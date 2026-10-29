"""The use-case bars U-a to U-d (decision note section 0.2), fixed before Z2 is read.

Inputs are ``gate_sim.seeded_runs`` results of three arms on the same pool, with the clean
labeler: the within-key partition, the production IVF refit, and the detector key alone. Nothing
here is tuned after a result.
"""

from __future__ import annotations

from typing import Any


U2_BAR = 0.95  # U-a, and the precision floor both arms must meet for U-c
IVF_FACTOR = 0.5  # U-b: decisions at most this share of IVF's
KEY_FACTOR = 0.9  # U-c: decisions at most this share of the detector-key arm's
OPEN_BAR = 0.90  # U-d
MIN_OPEN_PROPAGATED = 30  # U-d is not evaluable below this many propagated open-set items (mean)

PASS, FAIL, INCONCLUSIVE = 'PASS', 'FAIL', 'INCONCLUSIVE'


def _mean(block: dict[str, Any], key: str) -> float | None:
    cell = block.get(key)
    return None if cell is None else float(cell['mean'])


def _row(
    bar_id: str, metric: str, bar: float | None, value: float | None, result: str, note: str
) -> dict[str, Any]:
    return {
        'id': bar_id,
        'metric': metric,
        'bar': bar,
        'value': value,
        'result': result,
        'note': note,
    }


def u_bars(
    within: dict[str, Any], ivf: dict[str, Any], key_only: dict[str, Any]
) -> list[dict[str, Any]]:
    """U-a to U-d as rows with PASS, FAIL or INCONCLUSIVE and the limit that applies."""
    u2 = _mean(within, 'u2')
    u1 = _mean(within, 'u1')
    ivf_u1 = _mean(ivf, 'u1')
    key_u1 = _mean(key_only, 'u1')
    key_u2 = _mean(key_only, 'u2')
    ivf_u2 = _mean(ivf, 'u2')
    rows = []
    if u2 is None:
        rows.append(_row('U-a', 'u2', U2_BAR, None, INCONCLUSIVE, 'nothing was propagated'))
    else:
        rows.append(_row('U-a', 'u2', U2_BAR, u2, PASS if u2 >= U2_BAR else FAIL, ''))
    if u1 is None or ivf_u1 is None:
        rows.append(_row('U-b', 'u1_vs_ivf', None, u1, INCONCLUSIVE, 'u1 missing'))
    else:
        bar = IVF_FACTOR * ivf_u1
        rows.append(
            _row('U-b', 'u1_vs_ivf', bar, u1, PASS if u1 <= bar else FAIL, f'ivf u1 {ivf_u1:.4f}')
        )
    if u1 is None or key_u1 is None:
        rows.append(_row('U-c', 'u1_vs_key_only', None, u1, INCONCLUSIVE, 'u1 missing'))
    else:
        bar = KEY_FACTOR * key_u1
        both_floor = u2 is not None and u2 >= U2_BAR and key_u2 is not None and key_u2 >= U2_BAR
        if u2 is not None and u2 < U2_BAR:
            result, note = FAIL, 'the method misses the u2 floor'
        elif key_u2 is None or key_u2 < U2_BAR:
            result, note = (
                INCONCLUSIVE,
                (
                    'the detector-key arm is below the u2 floor, so there is no arm to compare '
                    'at equal precision'
                ),
            )
        else:
            result, note = (PASS if u1 <= bar else FAIL), ''
        rows.append(
            _row(
                'U-c',
                'u1_vs_key_only',
                bar,
                u1,
                result,
                f'key-only u1 {key_u1:.4f}, u2 {key_u2}; floor met by both: {both_floor}'
                + (f'; {note}' if note else ''),
            )
        )
    open_u2 = _mean(within, 'u2_open')
    open_n = _mean(within, 'open_propagated') or 0.0
    if open_u2 is None or open_n < MIN_OPEN_PROPAGATED:
        rows.append(
            _row(
                'U-d',
                'u2_open_set',
                OPEN_BAR,
                open_u2,
                INCONCLUSIVE,
                f'{open_n:.1f} open-set items propagated per run, fewer than {MIN_OPEN_PROPAGATED}',
            )
        )
    else:
        rows.append(
            _row(
                'U-d',
                'u2_open_set',
                OPEN_BAR,
                open_u2,
                PASS if open_u2 >= OPEN_BAR else FAIL,
                f'{open_n:.1f} open-set items propagated per run',
            )
        )
    rows[0]['ivf_u2'] = ivf_u2
    return rows


def overall(rows: list[dict[str, Any]]) -> str:
    """FAIL if any bar fails, else INCONCLUSIVE if any is, else PASS."""
    results = {r['result'] for r in rows}
    if FAIL in results:
        return FAIL
    return INCONCLUSIVE if INCONCLUSIVE in results else PASS


def simpler_winner(arms: dict[str, dict[str, Any]]) -> str | None:
    """The cheaper of IVF and the detector key alone among those at the u2 floor (the note's rule
    when a bar fails); ``None`` when neither reaches the floor."""
    ok: dict[str, float] = {}
    for name in ('ivf', 'key_only'):
        u1, u2 = _mean(arms[name], 'u1'), _mean(arms[name], 'u2')
        if u1 is not None and u2 is not None and u2 >= U2_BAR:
            ok[name] = u1
    return min(ok, key=lambda k: ok[k]) if ok else None
