"""DF6 guard: the `/embed/boxes` preprocessing comment must describe the
actual resampler used (`cv2.INTER_LINEAR` / bilinear), not LANCZOS.

`center_crop_cpu` (src/services/cpu_preprocess.py) has always used
`cv2.INTER_LINEAR`; the router comment claimed LANCZOS, which never matched
the code. This locks the comment to the true behavior so the two can't
silently drift apart again.
"""

from __future__ import annotations

from pathlib import Path


EMBED_ROUTER = Path(__file__).resolve().parent.parent / 'src' / 'routers' / 'embed.py'


def test_boxes_comment_does_not_claim_lanczos() -> None:
    text = EMBED_ROUTER.read_text()
    assert 'LANCZOS' not in text, (
        'src/routers/embed.py must not claim LANCZOS resampling for /embed/boxes '
        '(the actual crop resize is cv2.INTER_LINEAR / bilinear)'
    )


def test_boxes_comment_describes_bilinear() -> None:
    text = EMBED_ROUTER.read_text()
    assert 'BILINEAR' in text or 'INTER_LINEAR' in text
