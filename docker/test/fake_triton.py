"""A Triton stand-in for the live harness: liveness, readiness and an empty model repository.

The API refuses to start without a live Triton (src/app/startup_gate.py, owner decision D-69:
no degraded mode). The harness verifies the curation write paths without a GPU, so this answers
`ServerLive` and `ServerReady` over Triton's gRPC protocol, and over HTTP the readiness probe and
the repository index (an empty repository: the config vocabulary lists the served detectors from
it). Every other call stays unimplemented: a code path on the verified surface that needs
inference still fails loudly.
"""

from __future__ import annotations

import threading
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc


class _LiveOnly(service_pb2_grpc.GRPCInferenceServiceServicer):
    def ServerLive(  # noqa: N802 -- gRPC method name from Triton's proto
        self, _request: object, _context: object
    ) -> service_pb2.ServerLiveResponse:
        return service_pb2.ServerLiveResponse(live=True)

    def ServerReady(  # noqa: N802 -- gRPC method name from Triton's proto
        self, _request: object, _context: object
    ) -> service_pb2.ServerReadyResponse:
        return service_pb2.ServerReadyResponse(ready=True)


class _Http(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        ready = self.path == '/v2/health/ready'
        self._send(200 if ready else 404, b'{}')

    def do_POST(self) -> None:
        if self.path == '/v2/repository/index':
            self._send(200, b'[]')
        else:
            self._send(404, b'{}')

    def log_message(self, *_args: object) -> None:
        return None


def main() -> None:
    http = ThreadingHTTPServer(('0.0.0.0', 8000), _Http)  # nosec B104
    threading.Thread(target=http.serve_forever, daemon=True).start()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_LiveOnly(), server)
    server.add_insecure_port('0.0.0.0:8001')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
