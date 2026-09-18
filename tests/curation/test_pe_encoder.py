"""Unit tests for :mod:`src.clients.pe_encoder` (B-PR3).

The PE text encoder requires a PyTorch + ``perception_models`` checkpoint
load on first use. None of those heavy dependencies are exercised here —
we stub :meth:`PEEncoder.warm_text_encoder` to drop a fake torch model
into the instance, and stub the Triton pool with a ``MagicMock`` for the
image path. The tests assert the public-API contract only:

* image encoder returns 1024-d L2-normalized rows
* text encoder returns 1024-d L2-normalized rows
* LRU caching dedupes repeated queries
* the module imports cleanly without torch installed (covered by the
  fact that pytest collects this file without `import torch`)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.clients.pe_encoder import PE_EMBEDDING_DIM, PE_IMAGE_MODEL, PEEncoder, _l2_normalize


# =============================================================================
# Helpers
# =============================================================================


def _make_pool_returning(matrix: np.ndarray) -> MagicMock:
    """Build a fake AsyncTritonPool whose ``.infer`` returns ``matrix``."""
    pool = MagicMock()
    result = MagicMock()
    result.as_numpy = MagicMock(return_value=matrix)
    pool.infer = AsyncMock(return_value=result)
    return pool


def _stub_text_model(encoder: PEEncoder, vec: np.ndarray) -> None:
    """Pretend ``warm_text_encoder`` ran; install minimal mocks."""
    fake_torch = MagicMock()
    # torch.no_grad() context manager — must support __enter__/__exit__.
    fake_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    fake_features = MagicMock()
    fake_features.detach.return_value.cpu.return_value.numpy.return_value = vec.astype(np.float32)

    fake_model = MagicMock()
    fake_model.encode_text = MagicMock(return_value=fake_features)

    fake_tokenizer = MagicMock(return_value='TOKENS')

    encoder._torch = fake_torch
    encoder._text_model = fake_model
    encoder._text_tokenizer = fake_tokenizer
    encoder._text_ready = True


# =============================================================================
# Image path
# =============================================================================


@pytest.mark.asyncio
async def test_encode_images_returns_unit_norm_rows():
    raw = np.array(
        [
            [3.0] + [0.0] * (PE_EMBEDDING_DIM - 1),
            [0.0, 4.0] + [0.0] * (PE_EMBEDDING_DIM - 2),
        ],
        dtype=np.float32,
    )
    pool = _make_pool_returning(raw)
    enc = PEEncoder(triton_pool=pool)

    chws = np.zeros((2, 3, 16, 16), dtype=np.float32)
    out = await enc.encode_images(chws)

    assert out.shape == (2, PE_EMBEDDING_DIM)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0], atol=1e-6)
    # Triton was called against the expected model name.
    args, _kwargs = pool.infer.call_args
    assert args[0] == PE_IMAGE_MODEL


@pytest.mark.asyncio
async def test_encode_images_rejects_wrong_shape():
    enc = PEEncoder(triton_pool=_make_pool_returning(np.zeros((1, PE_EMBEDDING_DIM), np.float32)))
    with pytest.raises(ValueError, match='3, H, W'):
        await enc.encode_images(np.zeros((4, 16, 16), dtype=np.float32))


@pytest.mark.asyncio
async def test_encode_images_requires_triton_pool():
    enc = PEEncoder(triton_pool=None)
    with pytest.raises(RuntimeError, match='triton_pool'):
        await enc.encode_images(np.zeros((1, 3, 16, 16), dtype=np.float32))


# =============================================================================
# Text path
# =============================================================================


def test_encode_text_requires_warm():
    enc = PEEncoder()
    with pytest.raises(RuntimeError, match='warm_text_encoder'):
        enc.encode_text(['hello'])


def test_encode_text_returns_unit_norm():
    enc = PEEncoder()
    vec = np.array([3.0] + [0.0] * (PE_EMBEDDING_DIM - 1), dtype=np.float32)
    _stub_text_model(enc, vec)

    out = enc.encode_text(['red sedan'])

    assert out.shape == (1, PE_EMBEDDING_DIM)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-6)


def test_encode_text_lru_caches_repeats():
    enc = PEEncoder()
    vec = np.ones(PE_EMBEDDING_DIM, dtype=np.float32)
    _stub_text_model(enc, vec)

    enc.encode_text(['red sedan'])
    enc.encode_text(['red sedan'])
    enc.encode_text(['red sedan'])
    enc.encode_text(['blue truck'])

    info = enc.text_cache_info()
    # Two unique queries → two misses; three repeats → two hits (the
    # second + third call for 'red sedan').
    assert info.misses == 2
    assert info.hits == 2
    # Underlying torch model was only invoked once per unique query.
    assert enc._text_model.encode_text.call_count == 2


def test_encode_text_empty_returns_empty_matrix():
    enc = PEEncoder()
    out = enc.encode_text([])
    assert out.shape == (0, PE_EMBEDDING_DIM)
    assert out.dtype == np.float32


def test_text_ready_flag():
    enc = PEEncoder()
    assert enc.text_ready is False
    _stub_text_model(enc, np.ones(PE_EMBEDDING_DIM, dtype=np.float32))
    assert enc.text_ready is True


# =============================================================================
# Helper sanity
# =============================================================================


# =============================================================================
# Embedding-space sanity (P2-14) — image vs text encoder agreement
# =============================================================================


def test_encode_text_and_encode_images_share_embedding_space():
    """Both paths must land in the *same* 1024-d unit-norm space so a
    kNN query built from ``encode_text`` output can score against
    ``pe_embedding``/``v6_embedding`` docs written by the image path
    (the curation ingest pipeline's PE image-encoder Triton calls).

    This only asserts the *shape contract* (dim, dtype, L2-normalization)
    that :mod:`src.services.curation.semantic_search` depends on — it
    cannot exercise real cosine-similarity agreement between the actual
    PE-Core-L14-336 image tower and text tower without the real
    checkpoint, which requires ``torch`` + ``perception_models`` +
    downloaded weights, none of which are available in this sandbox.
    Skips gracefully (module already documents itself as importable
    without those deps — see this file's module docstring) rather than
    faking a pass.
    """
    try:
        import torch  # noqa: F401
        from core.vision_encoder import pe  # noqa: F401
    except ModuleNotFoundError as exc:
        pytest.skip(
            f'perception_models/torch checkpoint not available in this '
            f'sandbox ({exc}); cannot verify real image/text embedding-space '
            f'agreement here. Verified instead: both encode_images and '
            f'encode_text return L2-normalized (1024,)/(N, 1024) float32 '
            f'rows via the shape-contract tests above, which is the '
            f'contract semantic_search.py relies on.'
        )

    # If the real checkpoint *is* available, do a minimal same-space smoke
    # check: warm the text encoder for real and confirm it produces
    # unit-norm PE_EMBEDDING_DIM rows exactly like the mocked path above.
    enc = PEEncoder()
    enc.warm_text_encoder()
    out = enc.encode_text(['a photo of a white pickup truck'])
    assert out.shape == (1, PE_EMBEDDING_DIM)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-3)


def test_l2_normalize_handles_zero_rows():
    mat = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float32)
    out = _l2_normalize(mat)
    assert out.shape == (2, 3)
    # Zero row is left unchanged.
    np.testing.assert_array_equal(out[0], [0.0, 0.0, 0.0])
    # Non-zero row is normalized.
    np.testing.assert_allclose(np.linalg.norm(out[1]), 1.0, atol=1e-6)
