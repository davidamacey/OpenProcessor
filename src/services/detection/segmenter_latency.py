"""Measured latency of one full-image segmenter call, for the dry-run estimate.

The planning number used to be a fixed 3 s (the crop-stage figure); a full image
call on the live stack measured 0.28-0.35 s. Each call the API process makes
feeds an exponential moving average, so an estimate follows the hardware that
is actually serving. Per process and in memory: a process that has made no call
yet uses :data:`DEFAULT_SECONDS_PER_CALL`, the figure measured on the live stack.
"""

from __future__ import annotations


#: Seconds per full-image call before this process has measured one.
DEFAULT_SECONDS_PER_CALL = 0.36
#: Weight of the newest sample.
EMA_ALPHA = 0.2

_state: dict[str, float] = {}


def observe(seconds: float) -> None:
    prior = _state.get('ema')
    _state['ema'] = seconds if prior is None else EMA_ALPHA * seconds + (1 - EMA_ALPHA) * prior


def seconds_per_call() -> float:
    return _state.get('ema', DEFAULT_SECONDS_PER_CALL)


def reset() -> None:
    """Forget every sample (tests)."""
    _state.clear()


__all__ = ['DEFAULT_SECONDS_PER_CALL', 'EMA_ALPHA', 'observe', 'reset', 'seconds_per_call']
