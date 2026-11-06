"""A Triton stand-in for the live harness: it answers liveness and readiness and nothing else.

The API refuses to start without a live Triton (src/app/startup_gate.py, owner decision D-69:
no degraded mode). The harness verifies the curation write paths without a GPU, so this answers
`ServerLive` and `ServerReady` over Triton's gRPC protocol and leaves every other call
unimplemented: a code path on the verified surface that needs inference still fails loudly.
"""

from __future__ import annotations

from concurrent import futures

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


def main() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_LiveOnly(), server)
    server.add_insecure_port('0.0.0.0:8001')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
