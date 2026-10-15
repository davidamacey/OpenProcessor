"""Transport-only comparison for the gRPC ingest decision (#263, section 8 of
docs/design/ingest_surfaces_plan.md).

No pipeline, no GPU, no OpenSearch. A server process reads the bytes of each image, computes the
imohash the real route computes first, and returns a fixed-shape result like a
``BatchIngestResponse``. A client in a separate process sends the real bytes of an image
directory at a fixed client concurrency. Contenders:

* ``http1``: HTTP/1.1 multipart through uvicorn (uvloop, httptools) and FastAPI ``File`` parsing,
  the stack ``POST /ingest/upload`` runs on, one connection per client.
* ``grpc_stream``: ``grpc.aio`` client-streaming, an image is sent as chunks, one response per batch.
* ``grpc_unary``: ``grpc.aio`` unary, the whole batch in one message.

HTTP/2 multipart is not a contender here: it needs hypercorn or h2, which are not in the repo
venv, and the plan allows stating it as not measured.

The gRPC messages are hand-framed bytes (identity serializers, no protobuf codegen): a protobuf
``bytes`` field adds a tag and a length varint (about 5 bytes per message), which is below the
resolution of every number reported.

Per row, over ``--reps`` interleaved repetitions: images/s, request p50 and p99, server CPU and
client CPU seconds per image (``psutil`` process times, user plus system), bytes on the wire per
image (computed from the request and response sizes plus header and HTTP/2 framing estimates,
not captured from the network), peak resident memory of the server. The host's CPU busy percent
and load average are recorded per run so a noisy window is visible.

Two checks follow from ``--pipeline-cpu`` (API CPU seconds per image of the full pipeline, from
the baseline in docs/PERFORMANCE.md): the transport share (HTTP server CPU minus gRPC server
CPU, over the pipeline CPU) and the plan's skip rule (share of at least ``--threshold`` with
ranges that do not overlap).

    ingest_transport_bench.py --images <dir> --label coco --pipeline-cpu 0.18 \\
        --concurrency 4,16,32 --reps 3 --out transport_coco.json

Internal sub-commands ``serve-http`` and ``serve-grpc`` are what the runner spawns.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
from fastapi import FastAPI, File, Response, UploadFile


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CONTENDERS = ('http1', 'grpc_stream', 'grpc_unary')
CHUNK = 256 * 1024
MAX_MSG = 1024 * 1024 * 1024
HTTP2_FRAME = 16384
HTTP_HEADER_ALLOWANCE = 220  # request line, host, content-type, content-length, user-agent, accept
GRPC_CALL_HEADERS = 120  # :method, :path, content-type, te, grpc-timeout plus the trailers
_RESULT_ROW = {
    'status': 'success',
    'image_path': 'x' * 48,
    'image_id': 'f' * 32,
    'n_detections': 3,
    'n_embedded': 3,
    'error': None,
}


def _result_payload(count: int) -> bytes:
    """A response of the same shape and size class as a BatchIngestResponse."""
    import orjson

    rows = [dict(_RESULT_ROW) for _ in range(count)]
    return orjson.dumps({'results': rows, 'summary': {'successful': count, 'failed': 0}})


# --- servers ---------------------------------------------------------------------------------


def serve_http(port: int) -> None:
    import uvicorn

    from src.services.curation.ingest_helpers import _imohash_bytes

    app = FastAPI()

    @app.post('/ingest')
    async def ingest(images: list[UploadFile] = File(...)) -> Response:  # noqa: B008  # FastAPI File() default
        for upload in images:
            _imohash_bytes(await upload.read())
        return Response(_result_payload(len(images)), media_type='application/json')

    uvicorn.run(
        app,
        host='127.0.0.1',
        port=port,
        loop='uvloop',
        http='httptools',
        log_level='warning',
        access_log=False,
        backlog=4096,
    )


def serve_grpc(port: int) -> None:
    import grpc

    from src.services.curation.ingest_helpers import _imohash_bytes

    async def stream(request_iterator: Any, _context: Any) -> bytes:
        count = 0
        buf = bytearray()
        total = 0
        async for msg in request_iterator:
            if msg[0] == 1:
                total = int.from_bytes(msg[1:5], 'big')
                buf = bytearray(msg[5:])
            else:
                buf += msg[1:]
            if len(buf) == total:
                _imohash_bytes(bytes(buf))
                count += 1
        return _result_payload(count)

    async def unary(request: bytes, _context: Any) -> bytes:
        count = 0
        offset = 0
        while offset < len(request):
            size = int.from_bytes(request[offset : offset + 4], 'big')
            _imohash_bytes(request[offset + 4 : offset + 4 + size])
            offset += 4 + size
            count += 1
        return _result_payload(count)

    handler = grpc.method_handlers_generic_handler(
        'bench.Ingest',
        {
            'Stream': grpc.stream_unary_rpc_method_handler(stream),
            'Unary': grpc.unary_unary_rpc_method_handler(unary),
        },
    )

    async def main() -> None:
        server = grpc.aio.server(
            options=[
                ('grpc.max_receive_message_length', MAX_MSG),
                ('grpc.max_send_message_length', MAX_MSG),
            ]
        )
        server.add_generic_rpc_handlers((handler,))
        server.add_insecure_port(f'127.0.0.1:{port}')
        await server.start()
        await server.wait_for_termination()

    asyncio.run(main())


# --- client ----------------------------------------------------------------------------------


def _load(images: Path, limit: int, repeat: int = 1) -> list[bytes]:
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if limit:
        files = files[:limit]
    if not files:
        sys.exit(f'no images in {images}')
    return [p.read_bytes() for p in files] * repeat


def _batches(data: list[bytes], size: int) -> list[list[bytes]]:
    return [data[i : i + size] for i in range(0, len(data), size)]


def _frame_all(batch: list[bytes]) -> bytes:
    return b''.join(len(b).to_bytes(4, 'big') + b for b in batch)


def _stream_messages(batch: list[bytes]) -> list[bytes]:
    out: list[bytes] = []
    for img in batch:
        first = True
        for i in range(0, len(img), CHUNK):
            part = img[i : i + CHUNK]
            out.append((b'\x01' + len(img).to_bytes(4, 'big') if first else b'\x00') + part)
            first = False
    return out


def _grpc_wire(messages: list[bytes], response: int) -> int:
    payload = sum(5 + len(m) for m in messages) + 5 + response
    return payload + 9 * (payload // HTTP2_FRAME + 2) + GRPC_CALL_HEADERS


async def _run_client(
    name: str, port: int, batches: list[list[bytes]], concurrency: int
) -> tuple[list[float], int, int]:
    """Send every batch; returns request latencies, wire bytes and images."""
    queue: asyncio.Queue[list[bytes]] = asyncio.Queue()
    for b in batches:
        queue.put_nowait(b)
    lat: list[float] = []
    wire = 0

    if name == 'http1':
        import httpx

        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        async with httpx.AsyncClient(
            base_url=f'http://127.0.0.1:{port}', limits=limits, timeout=300.0
        ) as client:

            async def send(batch: list[bytes]) -> int:
                files = [('images', (f'i{k}.jpg', b, 'image/jpeg')) for k, b in enumerate(batch)]
                req = client.build_request('POST', '/ingest', files=files)
                resp = await client.send(req)
                resp.raise_for_status()
                return (
                    int(req.headers['content-length'])
                    + HTTP_HEADER_ALLOWANCE
                    + len(resp.content)
                    + sum(len(k) + len(v) + 4 for k, v in resp.headers.items())
                    + 17
                )

            wire = await _drain(queue, send, lat, concurrency)
    else:
        import grpc

        async with grpc.aio.insecure_channel(
            f'127.0.0.1:{port}',
            options=[
                ('grpc.max_receive_message_length', MAX_MSG),
                ('grpc.max_send_message_length', MAX_MSG),
            ],
        ) as channel:
            response_len = len(_result_payload(len(batches[0])))
            if name == 'grpc_stream':
                call = channel.stream_unary('/bench.Ingest/Stream')

                async def send(batch: list[bytes]) -> int:
                    msgs = _stream_messages(batch)
                    await call(_aiter(msgs))
                    return _grpc_wire(msgs, response_len)

            else:
                ucall = channel.unary_unary('/bench.Ingest/Unary')

                async def send(batch: list[bytes]) -> int:
                    body = _frame_all(batch)
                    await ucall(body)
                    return _grpc_wire([body], response_len)

            wire = await _drain(queue, send, lat, concurrency)
    return lat, wire, sum(len(x) for b in batches for x in b)


async def _aiter(items: list[bytes]) -> Any:
    for item in items:
        yield item


async def _drain(queue: Any, send: Any, lat: list[float], concurrency: int) -> int:
    total = [0]

    async def worker() -> None:
        while True:
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.perf_counter()
            sent = await send(batch)
            total[0] += sent
            lat.append(time.perf_counter() - t0)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return total[0]


# --- one measured run --------------------------------------------------------------------------


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _spawn(name: str, port: int) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        'serve-http' if name == 'http1' else 'serve-grpc',
        '--port',
        str(port),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosec B603  fixed argv list, no shell
    import socket

    for _ in range(200):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return proc
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f'{name} server did not start')


def _cpu(proc: psutil.Process) -> float:
    t = proc.cpu_times()
    return t.user + t.system


class _RssPeak(threading.Thread):
    def __init__(self, proc: psutil.Process) -> None:
        super().__init__(daemon=True)
        self.proc, self.peak, self.stop = proc, 0, threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except psutil.Error:
                return
            self.stop.wait(0.05)


def measure(name: str, data: list[bytes], batch_size: int, concurrency: int) -> dict[str, Any]:
    port = _free_port()
    server = _spawn(name, port)
    sp = psutil.Process(server.pid)
    me = psutil.Process()
    try:
        batches = _batches(data, batch_size)
        asyncio.run(_run_client(name, port, batches[: max(concurrency, 2)], concurrency))  # warm-up
        peak = _RssPeak(sp)
        peak.start()
        s0, c0 = _cpu(sp), _cpu(me)
        psutil.cpu_percent(None)
        load0 = os.getloadavg()[0]
        t0 = time.perf_counter()
        lat, wire, nbytes = asyncio.run(_run_client(name, port, batches, concurrency))
        wall = time.perf_counter() - t0
        host_busy = psutil.cpu_percent(None)
        s1, c1 = _cpu(sp), _cpu(me)
        peak.stop.set()
    finally:
        server.kill()
        server.wait()
    n = len(data)
    lat.sort()
    return {
        'contender': name,
        'concurrency': concurrency,
        'images': n,
        'image_mb': round(nbytes / 1e6, 1),
        'images_per_s': n / wall,
        'request_p50_s': statistics.median(lat),
        'request_p99_s': lat[min(len(lat) - 1, int(len(lat) * 0.99))],
        'server_cpu_s_per_image': (s1 - s0) / n,
        'client_cpu_s_per_image': (c1 - c0) / n,
        'wire_bytes_per_image': wire / n,
        'payload_bytes_per_image': nbytes / n,
        'server_peak_rss_mb': peak.peak / 1e6,
        'host_cpu_busy_pct': host_busy,
        'load_avg_start': load0,
    }


# --- report ------------------------------------------------------------------------------------


def _med(rows: list[dict[str, Any]], key: str) -> tuple[float, float, float]:
    vals = [r[key] for r in rows]
    return statistics.median(vals), min(vals), max(vals)


def summarize(rows: list[dict[str, Any]], pipeline_cpu: float, threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {'rows': [], 'decision': []}
    keys = sorted({(r['contender'], r['concurrency']) for r in rows})
    for name, conc in keys:
        sel = [r for r in rows if r['contender'] == name and r['concurrency'] == conc]
        entry: dict[str, Any] = {'contender': name, 'concurrency': conc, 'reps': len(sel)}
        for key in (
            'images_per_s',
            'request_p50_s',
            'request_p99_s',
            'server_cpu_s_per_image',
            'client_cpu_s_per_image',
            'wire_bytes_per_image',
            'server_peak_rss_mb',
            'host_cpu_busy_pct',
        ):
            entry[key] = _med(sel, key)
        out['rows'].append(entry)
    for conc in sorted({c for _, c in keys}):
        http = [r for r in rows if r['contender'] == 'http1' and r['concurrency'] == conc]
        for grpc_name in ('grpc_stream', 'grpc_unary'):
            g = [r for r in rows if r['contender'] == grpc_name and r['concurrency'] == conc]
            if not http or not g:
                continue
            h_cpu = [r['server_cpu_s_per_image'] for r in http]
            g_cpu = [r['server_cpu_s_per_image'] for r in g]
            saving = statistics.median(h_cpu) - statistics.median(g_cpu)
            out['decision'].append(
                {
                    'concurrency': conc,
                    'versus': grpc_name,
                    'server_cpu_saving_s_per_image': saving,
                    'transport_share': saving / pipeline_cpu,
                    'ranges_overlap': min(h_cpu) <= max(g_cpu),
                    'passes_rule': saving / pipeline_cpu >= threshold and min(h_cpu) > max(g_cpu),
                }
            )
    return out


def _fmt(entry: dict[str, Any], key: str, nd: int) -> str:
    m, lo, hi = entry[key]
    return f'{m:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})'


def print_table(label: str, summary: dict[str, Any]) -> None:
    print(f'\n{label}: median (min-max) of the repetitions')
    print(
        '| contender | clients | images/s | p50 s | p99 s | server CPU s/img | client CPU s/img | wire B/img | server RSS MB | host busy % |'
    )
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for e in summary['rows']:
        print(
            f'| {e["contender"]} | {e["concurrency"]} | {_fmt(e, "images_per_s", 1)} | {_fmt(e, "request_p50_s", 2)} | '
            f'{_fmt(e, "request_p99_s", 2)} | {_fmt(e, "server_cpu_s_per_image", 5)} | {_fmt(e, "client_cpu_s_per_image", 5)} | '
            f'{e["wire_bytes_per_image"][0]:.0f} | {e["server_peak_rss_mb"][0]:.0f} | {e["host_cpu_busy_pct"][0]:.0f} |'
        )
    print(
        '\n| clients | versus | server CPU saving s/img | share of pipeline CPU | ranges overlap | passes rule |'
    )
    print('|---:|---|---:|---:|---|---|')
    for d in summary['decision']:
        print(
            f'| {d["concurrency"]} | {d["versus"]} | {d["server_cpu_saving_s_per_image"]:.5f} | '
            f'{d["transport_share"] * 100:.2f}% | {d["ranges_overlap"]} | {d["passes_rule"]} |'
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='cmd')
    for name in ('serve-http', 'serve-grpc'):
        sub.add_parser(name).add_argument('--port', type=int, required=True)
    p.add_argument('--images', type=Path, help='directory of encoded images (the real bytes sent)')
    p.add_argument('--label', default='set', help='name of the image set in the report')
    p.add_argument('--limit', type=int, default=0, help='use only the first N images (0 = all)')
    p.add_argument('--repeat', type=int, default=1, help='send the image list N times (small sets)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--concurrency', default='4,16,32', help='comma list of client counts')
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--contenders', default=','.join(CONTENDERS))
    p.add_argument('--pipeline-cpu', type=float, help='full-pipeline API CPU s/image')
    p.add_argument('--threshold', type=float, default=0.10, help='skip rule: required share')
    p.add_argument('--out', type=Path, help='write the raw rows and the summary as JSON')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == 'serve-http':
        serve_http(args.port)
        return 0
    if args.cmd == 'serve-grpc':
        serve_grpc(args.port)
        return 0
    if args.images is None or args.pipeline_cpu is None:
        sys.exit('--images and --pipeline-cpu are required')
    data = _load(args.images, args.limit, args.repeat)
    names = args.contenders.split(',')
    unknown = set(names) - set(CONTENDERS)
    if unknown:
        sys.exit(f'unknown contenders {sorted(unknown)}; choose from {CONTENDERS}')
    rows: list[dict[str, Any]] = []
    for rep in range(args.reps):
        for conc in (int(c) for c in args.concurrency.split(',')):
            for name in names:
                row = measure(name, data, args.batch_size, conc)
                row['rep'] = rep
                rows.append(row)
                print(
                    f'rep {rep} {name} x{conc}: {row["images_per_s"]:.1f} img/s, server '
                    f'{row["server_cpu_s_per_image"]:.4f} s/img',
                    file=sys.stderr,
                )
    summary = summarize(rows, args.pipeline_cpu, args.threshold)
    print_table(args.label, summary)
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    'label': args.label,
                    'images': len(data),
                    'batch_size': args.batch_size,
                    'pipeline_cpu_s_per_image': args.pipeline_cpu,
                    'rows': rows,
                    'summary': summary,
                },
                indent=1,
            )
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
