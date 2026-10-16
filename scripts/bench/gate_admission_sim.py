"""Model of the per-process ingest admission gate under a closed loop of bulk-ingest requests.

``C`` requests are always in flight against ``P`` API worker processes. Each process admits a
request when its decoded megabytes fit under ``cap`` (the gate in ``src/utils/cpu_pool.py``) and
queues the rest first in, first out. A request of 16 photos of 20 MP is about 915 MiB, so under
the 1024 MiB default a second request on the same process waits for the first to finish. With
connections that stay on the worker that accepted them (``--sticky``), the busy processes are
fewer than the requests. The output is the throughput and the share of request time spent
waiting for the gate for several caps; the service time of an admitted request is a lognormal
around ``--service-s``. It is a model of queue structure, not a measurement of the stack.

    gate_admission_sim.py --processes 24 --requests 16 --request-mib 915
"""

from __future__ import annotations

import argparse
import heapq
import random
import statistics


def simulate(
    *,
    processes: int,
    requests: int,
    request_mib: int,
    cap: int,
    sticky: bool,
    service_s: float,
    horizon_s: float,
    seed: int,
    images_per_request: int = 16,
) -> tuple[float, float]:
    """(images per second, share of request time spent waiting for the gate)."""
    rng = random.Random(seed)
    charge = min(request_mib, cap)
    free = [cap] * processes
    queues: list[list[tuple[int, float]]] = [[] for _ in range(processes)]
    conn = [rng.randrange(processes) for _ in range(requests)]
    events: list[tuple[float, int, int, float]] = []
    waited = 0.0
    total = 0.0
    done = 0

    def admit(p: int, now: float) -> None:
        nonlocal waited
        while queues[p] and free[p] >= charge:
            client, queued_at = queues[p].pop(0)
            free[p] -= charge
            waited += now - queued_at
            span = rng.lognormvariate(0, 0.3) * service_s
            heapq.heappush(events, (now + span, client, p, queued_at))

    def send(client: int, now: float) -> None:
        p = conn[client] if sticky else rng.randrange(processes)
        queues[p].append((client, now))
        admit(p, now)

    for client in range(requests):
        send(client, 0.0)
    while events:
        now, client, p, queued_at = heapq.heappop(events)
        if now > horizon_s:
            break
        free[p] += charge
        done += 1
        total += now - queued_at
        admit(p, now)
        send(client, now)
    return done * images_per_request / horizon_s, waited / max(total, 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--processes', type=int, default=24)
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--request-mib', type=int, default=915)
    parser.add_argument('--service-s', type=float, default=20.0)
    parser.add_argument('--horizon-s', type=float, default=20000.0)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()
    for cap in (1024, 2048, 4096):
        for sticky in (False, True):
            rows = [
                simulate(
                    processes=args.processes,
                    requests=args.requests,
                    request_mib=args.request_mib,
                    cap=cap,
                    sticky=sticky,
                    service_s=args.service_s,
                    horizon_s=args.horizon_s,
                    seed=seed,
                )
                for seed in range(args.runs)
            ]
            rate = statistics.mean(r[0] for r in rows)
            share = statistics.mean(r[1] for r in rows)
            label = 'sticky' if sticky else 'random'
            print(
                f'cap {cap:5d} MiB {label}: {rate:5.1f} images/s, waiting {share:.2f} of request time'
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
