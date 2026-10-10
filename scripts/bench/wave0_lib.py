"""Pure logic for the Wave 0 full-pipeline probes: Triton config rewrites, host CPU share,
GPU busy share from Triton statistics, py-spy stack aggregation, sweep bookkeeping.

No network, no subprocess, no filesystem: everything the Wave 0 tables report goes through a
function here that a unit test pins down. The probes that touch the machine are in
``wave0_probes.py`` and the phases in ``wave0_suite.py``.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


# ---------------------------------------------------------------- Triton model config rewrites

_COUNT = re.compile(r'(\bcount\s*:\s*)\d+')
_GPUS = re.compile(r'(\bgpus\s*:\s*)\[[^\]]*\]')
_DELAY = re.compile(r'(\bmax_queue_delay_microseconds\s*:\s*)\d+')
_PREFERRED = re.compile(r'^[ \t]*preferred_batch_size\s*:\s*\[[^\]]*\][ \t]*\n', re.MULTILINE)


def _instance_group_span(text: str) -> tuple[int, int]:
    """Start and end offsets of the ``instance_group [ ... ]`` block (brackets included)."""
    head = re.search(r'\binstance_group\s*\[', text)
    if head is None:
        msg = 'config has no instance_group block'
        raise ValueError(msg)
    depth, i = 0, head.end() - 1
    while i < len(text):
        depth += (text[i] == '[') - (text[i] == ']')
        if depth == 0:
            return head.start(), i + 1
        i += 1
    msg = 'unbalanced instance_group block'
    raise ValueError(msg)


def rewrite_instance_group(
    text: str, *, count: int | None = None, gpus: list[int] | None = None
) -> str:
    """Set ``count`` and/or ``gpus`` of the (single) instance group of a model config."""
    start, end = _instance_group_span(text)
    block = text[start:end]
    if count is not None:
        block = _COUNT.sub(lambda m: f'{m[1]}{count}', block, count=1)
    if gpus is not None:
        block = _GPUS.sub(lambda m: f'{m[1]}[ {", ".join(str(g) for g in gpus)} ]', block, count=1)
    return text[:start] + block + text[end:]


def rewrite_batching(
    text: str, *, queue_delay_us: int | None = None, drop_preferred: bool = False
) -> str:
    """Set ``max_queue_delay_microseconds`` and/or remove ``preferred_batch_size``."""
    if queue_delay_us is not None:
        text = _DELAY.sub(lambda m: f'{m[1]}{queue_delay_us}', text, count=1)
    if drop_preferred:
        text = _PREFERRED.sub('', text, count=1)
    return text


def parse_arm(spec: str) -> dict[str, int]:
    """``'pe=2,det=3'`` -> ``{'pe': 2, 'det': 3}``; an empty spec is the shipped configuration.

    Keys are a model key, optionally with a ``_delay`` or ``_nopref`` knob (``pe_delay=2000``).
    """
    out: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in spec.split(','))):
        key, _, value = part.partition('=')
        if not value.isdigit():
            msg = f'arm entry {part!r} must be name=count'
            raise ValueError(msg)
        out[key] = int(value)
    return out


def parse_points(spec: str) -> list[tuple[int, int]]:
    """``'32x4,8x16'`` -> ``[(32, 4), (8, 16)]`` as (batch size, client concurrency)."""
    points = []
    for part in filter(None, (p.strip() for p in spec.split(','))):
        batch, _, conc = part.partition('x')
        if not (batch.isdigit() and conc.isdigit()):
            msg = f'point {part!r} must be <batch>x<concurrency>'
            raise ValueError(msg)
        points.append((int(batch), int(conc)))
    return points


# ---------------------------------------------------------------- host and GPU shares


def parse_proc_stat(text: str) -> tuple[int, int]:
    """(busy jiffies, total jiffies) from the aggregate ``cpu`` line of ``/proc/stat``."""
    fields = [int(x) for x in text.splitlines()[0].split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
    total = sum(fields[:8])  # user nice system idle iowait irq softirq steal
    return total - idle, total


def host_busy_pct(before: tuple[int, int], after: tuple[int, int]) -> float:
    """Whole-host CPU busy share over a window, percent."""
    busy, total = after[0] - before[0], after[1] - before[1]
    return 100.0 * busy / total if total > 0 else 0.0


def triton_gpu_busy(delta: dict[str, dict[str, Any]], wall_s: float) -> dict[str, float]:
    """Seconds of model execution per model and in total, as a share of wall time.

    ``delta`` is :func:`suite_lib.triton_stats_delta` output; the share can exceed 1.0 when
    instances overlap on the GPU, so it is an upper bound of "busy" and a lower bound of load.
    """
    per_model = {
        m: d['exec_infer_ms'] * d['executions'] / 1000.0 / wall_s for m, d in delta.items()
    }
    return {**{f'busy_{m}': v for m, v in per_model.items()}, 'busy_total': sum(per_model.values())}


def clock_summary(samples: list[dict[str, float]]) -> dict[str, float]:
    """Mean/min SM clock, mean power and the share of samples with a throttle reason set."""
    if not samples:
        return {}
    clocks = [s['sm_mhz'] for s in samples]
    return {
        'samples': len(samples),
        'sm_mhz_mean': sum(clocks) / len(clocks),
        'sm_mhz_min': min(clocks),
        'power_w_mean': sum(s['power_w'] for s in samples) / len(samples),
        'power_w_max': max(s['power_w'] for s in samples),
        'throttled_share': sum(1 for s in samples if s['throttled']) / len(samples),
    }


def parse_clock_line(line: str) -> dict[str, float] | None:
    """One ``nvidia-smi --query-gpu=index,clocks.sm,power.draw,clocks_throttle_reasons.active``
    CSV line (no units) -> ``{'gpu', 'sm_mhz', 'power_w', 'throttled'}`` or ``None``.

    A throttle mask of 0x0 or 0x1 (GPU idle) is not throttling; any other bit counts.
    """
    parts = [p.strip() for p in line.split(',')]
    if len(parts) != 4:
        return None
    try:
        mask = int(parts[3], 16)
        return {
            'gpu': float(parts[0]),
            'sm_mhz': float(parts[1]),
            'power_w': float(parts[2]),
            'throttled': float(mask & ~0x1 != 0),
        }
    except ValueError:
        return None


# ---------------------------------------------------------------- py-spy attribution

BUCKETS = (
    ('image_decode_encode', ('cv2', 'PIL', 'imdecode', 'imencode', 'turbojpeg')),
    ('numpy_tensor_build', ('numpy', 'preprocess', 'letterbox', 'normalize')),
    ('triton_client_grpc', ('tritonclient', 'grpc', 'triton_pool', 'triton_client')),
    ('opensearch_client', ('opensearchpy', 'opensearch', 'urllib3', 'bulk')),
    ('json_serialization', ('json', 'serializer', 'orjson', 'pydantic')),
    ('event_loop_wait', ('selectors', 'asyncio', 'uvloop', 'anyio')),
)


def classify_frame(frame: str) -> str:
    """Bucket of one py-spy frame ``'func (path:line)'`` by function and module path."""
    low = frame.lower()
    for name, needles in BUCKETS:
        if any(n.lower() in low for n in needles):
            return name
    return 'other_python'


def aggregate_collapsed(lines: list[str]) -> dict[str, Any]:
    """Aggregate py-spy ``--format raw`` lines (``frame;frame;... count``).

    Returns total samples, the leaf-frame buckets (where the CPU is when sampled), and the top
    leaf and inclusive frames. Each non-idle sample is one thread that was on CPU or holding the
    GIL, so the counts are proportional to CPU time of the Python processes.
    """
    leaf_buckets: Counter[str] = Counter()
    leaf: Counter[str] = Counter()
    inclusive: Counter[str] = Counter()
    total = 0
    for line in lines:
        stack, _, count = line.rstrip().rpartition(' ')
        if not stack or not count.isdigit():
            continue
        n = int(count)
        frames = stack.split(';')
        total += n
        leaf[frames[-1]] += n
        leaf_buckets[classify_frame(frames[-1])] += n
        for frame in set(frames):
            inclusive[frame] += n
    return {
        'samples': total,
        'leaf_bucket_share': {k: v / total for k, v in leaf_buckets.most_common()} if total else {},
        'top_leaf': [(f, c / total) for f, c in leaf.most_common(25)] if total else [],
        'top_inclusive': [(f, c / total) for f, c in inclusive.most_common(40)] if total else [],
    }


# ---------------------------------------------------------------- binding resource


def binding_resource(
    *, gpu_busy_pct: float, host_busy_pct_: float, api_cores: float, api_cores_available: float
) -> str:
    """One-word reading of a run: ``gpu``, ``api-cpu``, ``host-cpu`` or ``latency/feeding``.

    GPU busy of 90 percent or more binds on the GPU; an API at 90 percent of the cores it can use
    or a host at 90 percent busy binds on the host; otherwise the pipeline is neither saturated
    (clients, batching or serial sections feed it too slowly).
    """
    if gpu_busy_pct >= 90.0:
        return 'gpu'
    if api_cores_available > 0 and api_cores / api_cores_available >= 0.9:
        return 'api-cpu'
    if host_busy_pct_ >= 90.0:
        return 'host-cpu'
    return 'latency/feeding'
