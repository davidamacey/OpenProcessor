"""Deterministic stand-in for the segmenter service (``docker/segmenter``).

Answers the two routes the API calls (``GET /health``, ``POST /segment``) with a
fixed set of wheel-shaped candidates in the submitted crop's normalized frame,
so a region profile can be activated and "Test on a crop" can run on a stack
with no GPU. The candidates are synthetic geometry, not segmentation output:
what is real is the route, the candidate selection and the rendering of the
result. Standard library only.
"""

from __future__ import annotations

import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PORT = int(os.environ.get('FAKE_SEGMENTER_PORT', '8000'))

# (bbox_norm in the crop frame, score): two wheels of a side-on car and a weak
# bumper hit that a profile's score and count limits are expected to drop.
CANDIDATES = (
    ((0.10, 0.60, 0.30, 0.95), 0.91),
    ((0.65, 0.60, 0.90, 0.95), 0.86),
    ((0.45, 0.55, 0.55, 0.65), 0.22),
)


def ellipse(box: tuple[float, float, float, float], points: int = 24) -> list[list[float]]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    rx, ry = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
    return [
        [
            round(cx + rx * math.cos(2 * math.pi * i / points), 4),
            round(cy + ry * math.sin(2 * math.pi * i / points), 4),
        ]
        for i in range(points)
    ]


def segment(request: dict) -> dict:
    limit = int(request.get('max_candidates') or len(CANDIDATES))
    floor = request.get('min_score')
    masks = bool(request.get('return_masks'))
    found = [
        {
            'bbox_norm': list(box),
            'score': score,
            'mask_iou': 0.78 if masks else None,
            'mask_polygon': ellipse(box) if masks else None,
        }
        for box, score in CANDIDATES
        if floor is None or score >= float(floor)
    ]
    return {
        'candidates': found[:limit],
        'elapsed_ms': 1.0,
        'crop_size': [0, 0],
        'prompt': request.get('text_prompt', ''),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # http.server hook name
        if self.path == '/health':
            self._send(
                200,
                {
                    'status': 'healthy',
                    'model': 'sam3',
                    'device': 'cpu',
                    'loaded': True,
                    'instances': 1,
                    'max_candidates': 4,
                    'default_min_score': 0.5,
                },
            )
        else:
            self._send(404, {'detail': 'not found'})

    def do_POST(self) -> None:  # http.server hook name
        length = int(self.headers.get('Content-Length') or 0)
        body = json.loads(self.rfile.read(length) or b'{}')
        if self.path == '/segment':
            self._send(200, segment(body))
        else:
            self._send(404, {'detail': 'not found'})

    def log_message(self, *_args: object) -> None:
        return


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', PORT), Handler).serve_forever()  # nosec B104  # container-internal harness server
